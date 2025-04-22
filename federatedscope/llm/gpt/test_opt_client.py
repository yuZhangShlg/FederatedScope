import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_utils import get_parameter_dtype

from test_opt_server import ServerGPT2Model


class GradientStorage:
    def __init__(self):
        self.grad = []

    def __call__(self, grad):
        self.grad.append(grad)

    def reset(self):
        self.grad = []

    def index_grad(self, index):
        return self.grad[index]

    def __len__(self):
        return len(self.grad)

    def index_change_grad(self, index, grad):
        self.grad[index] = grad

    def sum_final_grad(self):
        self.grad[-2] += self.grad[-1]
        self.grad.pop(-1)


class GraphValueStorage:
    def __init__(self):
        self.graph_value = []

    def __call__(self, graph_value):
        self.graph_value.append(graph_value)

    def reset(self):
        self.graph_value = []

    def index_graph_value(self, index):
        return self.graph_value[index]

    def __len__(self):
        return len(self.graph_value)


class ClientLoraModel(nn.Module):
    def __init__(self, original_attention):
        super().__init__()
        self.ln_1 = copy.deepcopy(original_attention.ln_1)
        self.lora_A = copy.deepcopy(original_attention.lora_A)
        self.lora_B = copy.deepcopy(original_attention.lora_B)
        self.lora_dropout = copy.deepcopy(original_attention.lora_dropout)

    def forward(self, input_ids):
        input_ids = self.ln_1(input_ids)
        input_ids = self.lora_B(self.lora_A(input_ids))
        return self.lora_dropout(input_ids)


class ClientGPT2Model(nn.Module):
    def __init__(self, original_transformer):
        super().__init__()

        self.config = original_transformer.config

        self.wte = copy.deepcopy(original_transformer.wte)
        self.wpe = copy.deepcopy(original_transformer.wpe)
        self.drop = copy.deepcopy(original_transformer.drop)
        self.h = copy.deepcopy(original_transformer.h[:1])
        self.lora_attn = nn.ModuleList(
            [ClientLoraModel(attn) for attn in original_transformer.h[1:]])
        self.ln_f = copy.deepcopy(original_transformer.ln_f)

        self.gradient_checkpointing = original_transformer.gradient_checkpointing

    @property
    def dtype(self):
        return get_parameter_dtype(self)

    def get_head_mask(
            self, head_mask, num_hidden_layers: int, is_attention_chunked: bool = False
    ):
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)  # switch to float if need + fp16 compatibility
        return head_mask

    def forward(self, input_ids, model_index, **kwargs):
        if model_index == -1:
            hidden_states, output_shape = self.first_block_forward(input_ids, **kwargs)

            return hidden_states, output_shape
        else:
            return self.lora_attn[model_index](input_ids)

    def first_block_forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, use_cache, output_attentions)

                return custom_forward

            outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.h[0]),
                hidden_states,
                None,
                attention_mask,
                head_mask[0],
                encoder_hidden_states,
                encoder_attention_mask,
            )
        else:
            outputs = self.h[0](
                hidden_states,
                layer_past=past_key_values[0],
                attention_mask=attention_mask,
                head_mask=head_mask[0],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
        hidden_states = outputs[0]

        return hidden_states, output_shape


class ClientGPT2HeadModel(nn.Module):
    def __init__(self, client_model, server_model):
        super().__init__()
        self.client_transformer = ClientGPT2Model(client_model.transformer)
        self.lm_head = copy.deepcopy(client_model.lm_head)
        self.server_transformer = ServerGPT2Model(server_model.transformer)
        self.config = copy.deepcopy(client_model.config)

        self.client_storage_grad = GradientStorage()
        self.client_storage_graph = GraphValueStorage()
        self.client_storage_grad_s = GradientStorage()
        self.client_storage_graph_s = GraphValueStorage()

        self.server_storage_grad = GradientStorage()
        self.server_storage_graph = GraphValueStorage()

    def forward(self, input_ids, labels, **kwargs):
        self.client_storage_grad.reset()
        self.client_storage_grad_s.reset()
        self.client_storage_graph.reset()
        self.client_storage_graph_s.reset()
        self.server_storage_grad.reset()
        self.server_storage_graph.reset()

        return_dict = kwargs.get("return_dict", None)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states, output_shape = self.client_transformer(
            input_ids, model_index=-1, **kwargs
        )
        self.client_storage_graph_s(hidden_states)

        hidden_states = hidden_states.detach().cpu().numpy()
        hidden_states = torch.from_numpy(hidden_states)
        hidden_states.requires_grad_(True)
        hidden_states.register_hook(self.client_storage_grad_s)

        for idx in range(len(self.client_transformer.lora_attn)):
            hidden_states_middle = self.client_transformer(hidden_states, idx, **kwargs)
            self.server_storage_graph(hidden_states_middle)

            hidden_states_middle = hidden_states_middle.detach().cpu().numpy()
            hidden_states_middle = torch.from_numpy(hidden_states_middle)
            hidden_states_middle.requires_grad_(True)
            hidden_states_middle.register_hook(self.server_storage_grad)

            hidden_states = self.server_transformer(hidden_states, hidden_states_middle, idx, **kwargs)
            self.client_storage_graph(hidden_states)

            hidden_states = hidden_states.detach().cpu().numpy()
            hidden_states = torch.from_numpy(hidden_states)
            hidden_states.requires_grad_(True)
            hidden_states.register_hook(self.client_storage_grad)

        hidden_states = hidden_states.view(output_shape)
        hidden_states = BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )

        hidden_result = hidden_states[0]
        lm_logits = self.lm_head(hidden_result)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=hidden_states.past_key_values,
            hidden_states=hidden_states.hidden_states,
            attentions=hidden_states.attentions,
            cross_attentions=hidden_states.cross_attentions,
        )

    def backward(self, loss):
        loss.backward(retain_graph=True)
        _len = len(self.client_transformer.lora_attn)

        for idx in range(len(self.client_transformer.lora_attn)):
            torch.autograd.backward(self.client_storage_graph.index_graph_value(_len - idx - 1),
                                    self.client_storage_grad.index_grad(-1),
                                    retain_graph=True)

            torch.autograd.backward(self.server_storage_graph.index_graph_value(_len - idx - 1),
                                    self.server_storage_grad.index_grad(-1),
                                    retain_graph=True)
            if idx < _len - 1:
                self.client_storage_grad.sum_final_grad()

        self.client_storage_grad_s.sum_final_grad()
        torch.autograd.backward(self.client_storage_graph_s.index_graph_value(0),
                                self.client_storage_grad_s.index_grad(0),
                                retain_graph=True)

