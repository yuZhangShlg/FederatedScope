
import copy
from typing import Optional, List, Union, Tuple

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2PreTrainedModel
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import (
    logging,
    add_start_docstrings,
    replace_return_docstrings,
    add_start_docstrings_to_model_forward
)
from .utils import QWEN2_START_DOCSTRING, QWEN2_INPUTS_DOCSTRING


logger = logging.get_logger(__name__)


_CONFIG_FOR_DOC = "Qwen2Config"


class ServerLoraModel(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.scaling = config.scaling

        self.input_layernorm = copy.deepcopy(model.input_layernorm)

        self.q_proj_lora_A = copy.deepcopy(model.q_proj_lora_A)
        self.q_proj_lora_B = copy.deepcopy(model.q_proj_lora_B)
        self.q_proj_lora_dropout = copy.deepcopy(model.q_proj_lora_dropout)

        self.k_proj_lora_A = copy.deepcopy(model.k_proj_lora_A)
        self.k_proj_lora_B = copy.deepcopy(model.k_proj_lora_B)
        self.k_proj_lora_dropout = copy.deepcopy(model.k_proj_lora_dropout)

        self.v_proj_lora_A = copy.deepcopy(model.v_proj_lora_A)
        self.v_proj_lora_B = copy.deepcopy(model.v_proj_lora_B)
        self.v_proj_lora_dropout = copy.deepcopy(model.v_proj_lora_dropout)

    def forward(self, input_ids):
        input_ids = self.input_layernorm(input_ids)

        q_proj = self.q_proj_lora_B(self.q_proj_lora_A(
            self.q_proj_lora_dropout(input_ids))) * self.scaling

        k_proj = self.k_proj_lora_B(self.k_proj_lora_A(
            self.k_proj_lora_dropout(input_ids))) * self.scaling

        v_proj = self.v_proj_lora_B(self.v_proj_lora_A(
            self.v_proj_lora_dropout(input_ids))) * self.scaling

        return q_proj, k_proj, v_proj


@add_start_docstrings(
    "The bare Qwen2 Model outputting raw hidden-states without any specific head on top.",
    QWEN2_START_DOCSTRING,
)
class ServerQwen2Model(Qwen2PreTrainedModel):

    def __init__(self, model, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self._attn_implementation = config._attn_implementation

        self.embed_tokens = model.embed_tokens
        self.layers = copy.deepcopy(model.layers[:1])
        self.lora_attn = nn.ModuleList(
            [ServerLoraModel(attn, config) for attn in model.layers[1:]])
        self.gradient_checkpointing = model.gradient_checkpointing

    def forward(self, input_ids, model_index, **kwargs):
        if model_index == -1:
            hidden_states, output_shape = self.first_block_forward(input_ids, **kwargs)
            return hidden_states, output_shape
        else:
            return self.lora_attn[model_index](input_ids)

    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    def first_block_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        hidden_states = inputs_embeds
        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                self.layers[0].__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
            )
        else:
            layer_outputs = self.layers[0](
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]
        return hidden_states, None


class ServerQwen2ForCausalLM(Qwen2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, model):
        super().__init__(model.config)
        config = model.config
        self.model = ServerQwen2Model(model.model, config)
        self.lm_head = copy.deepcopy(model.lm_head)
        self.vocab_size = config.vocab_size

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        model_index: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            model_index=model_index,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return outputs

    def tail_forward(self, outputs, labels, **kwargs):
        return_dict = kwargs.get("return_dict", None)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
