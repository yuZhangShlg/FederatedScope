import copy

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from transformers import GPT2LMHeadModel
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from peft import LoraConfig, get_peft_model, TaskType
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


def change_model_layer():
    _path = "/bigdata/zy/FederatedScope/model/gpt2"
    model = AutoModelForCausalLM.from_pretrained(_path)

    args = {'r': 8, 'lora_alpha': 32, 'lora_dropout': 0.1, "fan_in_fan_out": True}
    peft_config = LoraConfig(task_type="CAUSAL_LM", **args)
    model = get_peft_model(model, peft_config)

    for i in range(len(model.transformer.h)):
        original_block = model.transformer.h[i]
        model.transformer.h[i] = ModifiedGPT2Block(original_block)

    return model


class ModifiedGPT2Block(nn.Module):
    def __init__(self, original_block):
        super().__init__()
        self.ln_1 = copy.deepcopy(original_block.ln_1)
        self.attn = copy.deepcopy(original_block.attn)
        self.ln_2 = copy.deepcopy(original_block.ln_2)
        self.mlp = copy.deepcopy(original_block.mlp)

        if hasattr(original_block, "crossattention"):
            self.crossattention = copy.deepcopy(original_block.crossattention)
        if hasattr(original_block, "ln_cross_attn"):
            self.ln_cross_attn = copy.deepcopy(original_block.ln_cross_attn)

    def forward(self, hidden_states, **kwargs):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=kwargs.get('layer_past', None),
            attention_mask=kwargs.get('attention_mask', None),
            head_mask=kwargs.get('head_mask', None),
            use_cache=kwargs.get('use_cache', False),
            output_attentions=kwargs.get('output_attentions', False),
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if kwargs.get('encoder_hidden_states', None) is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "cross attention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=kwargs.get('attention_mask', None),
                head_mask=kwargs.get('head_mask', None),
                encoder_hidden_states=kwargs.get('encoder_hidden_states', None),
                encoder_attention_mask=kwargs.get('encoder_attention_mask', None),
                output_attentions=kwargs.get('output_attentions', False),
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if kwargs.get('use_cache', False):
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


if __name__ == '__main__':
    _data = torch.tensor([21106, 318, 281, 12064, 326, 8477, 257, 4876, 13, 19430,
                          257, 2882, 326, 20431, 32543, 262, 2581, 13, 198, 198,
                          21017, 46486, 25, 198, 23318, 1115, 9040, 329, 10589, 5448,
                          13, 198, 198, 21017, 18261, 25, 16, 13, 47659, 257,
                          12974, 5496, 290, 787, 1654, 284, 2291, 6088, 286, 15921,
                          290, 13701, 13, 220, 198, 17, 13, 32900, 7987, 284,
                          1394, 534, 1767, 4075, 290, 1913, 13, 220, 198, 18,
                          13, 3497, 1576, 3993, 290, 5529, 257, 6414, 3993, 7269,
                          13, 50256])
    _label = torch.tensor([-100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                           -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                           -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                           -100, -100, -100, -100, -100, -100, 16, 13, 47659, 257,
                           12974, 5496, 290, 787, 1654, 284, 2291, 6088, 286, 15921,
                           290, 13701, 13, 220, 198, 17, 13, 32900, 7987, 284,
                           1394, 534, 1767, 4075, 290, 1913, 13, 220, 198, 18,
                           13, 3497, 1576, 3993, 290, 5529, 257, 6414, 3993, 7269,
                           13, 50256])
    model = change_model_layer()
    result = model(_data, labels=_label)
    print(f"loss: {result[0]}")
    print(f"result shape: {result[1].shape}")
    print(f"result: {result[1]}")
