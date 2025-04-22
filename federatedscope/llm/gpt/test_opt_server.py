import copy

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from transformers.utils import add_start_docstrings
from transformers.utils.model_parallel_utils import get_device_map, assert_device_map
from transformers.modeling_utils import get_parameter_dtype


class ServerGPT2Attention(nn.Module):

    def __init__(self, original_attention):
        super().__init__()
        self.c_attn = copy.deepcopy(original_attention.c_attn)
        self.c_proj = copy.deepcopy(original_attention.c_proj)
        self.attn_dropout = copy.deepcopy(original_attention.attn_dropout)
        self.resid_dropout = copy.deepcopy(original_attention.resid_dropout)
        self.bias = copy.deepcopy(original_attention.bias)

        self.num_heads = original_attention.num_heads
        self.head_dim = original_attention.head_dim
        self.split_size = original_attention.split_size
        self.layer_idx = original_attention.layer_idx

        self.scale_attn_weights = original_attention.scale_attn_weights
        self.scale_attn_by_inverse_layer_idx = original_attention.scale_attn_by_inverse_layer_idx
        self.is_cross_attention = original_attention.is_cross_attention

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def forward(self, hidden_states, lora_state, **kwargs):
        layer_past = kwargs.get('layer_past', None)
        use_cache = kwargs.get('use_cache', False)
        attention_mask = kwargs.get('attention_mask', None)
        head_mask = kwargs.get('head_mask', None)
        output_attentions = kwargs.get('output_attentions', False)

        lora_state = lora_state.to(hidden_states.device)
        query, key, value = (self.c_attn(hidden_states) + lora_state).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(
            query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class ServerGPT2Block(nn.Module):
    def __init__(self, original_block):
        super().__init__()
        self.attn = ServerGPT2Attention(original_block.attn)

        self.ln_1 = copy.deepcopy(original_block.ln_1)
        self.ln_2 = copy.deepcopy(original_block.ln_2)
        self.mlp = copy.deepcopy(original_block.mlp)

        del original_block.ln_1

    def forward(self, hidden_states, lora_state, **kwargs):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            lora_state,
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


class ServerGPT2Model(nn.Module):
    _keys_to_ignore_on_load_missing = ["attn.masked_bias"]

    def __init__(self, original_transformer):
        super().__init__()

        self.h = nn.ModuleList(
            [ServerGPT2Block(original_transformer.h[i]) for i in range(len(original_transformer.h))]
        )
        self.ln_f = copy.deepcopy(original_transformer.ln_f)

        # Model parallel
        self.model_parallel = original_transformer.model_parallel
        self.device_map = original_transformer.device_map
        self.gradient_checkpointing = original_transformer.gradient_checkpointing
        self.config = original_transformer.config

        # model param
        self.batch_size = None
        self.output_shape = None
        self.past_key_values = None
        self.output_attentions = None
        self.attention_mask = None
        self.head_mask = None
        self.use_cache = None

    def parallelize(self, device_map=None):

        self.device_map = (
            get_device_map(len(self.h), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.h))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))

        # Load onto devices
        for k, v in self.device_map.items():
            for block in v:
                cuda_device = "cuda:" + str(k)
                self.h[block] = self.h[block].to(cuda_device)

        # ln_f to last
        self.ln_f = self.ln_f.to(self.last_device)

    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        for index in range(len(self.h)):
            self.h[index] = self.h[index].to("cpu")
        self.ln_f = self.ln_f.to("cpu")
        torch.cuda.empty_cache()

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
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)  # switch to float if need + fp16 compatibility
        return head_mask

    def init_param(self, hidden_states, **kwargs):
        use_cache = kwargs.get('use_cache')
        output_attentions = kwargs.get('output_attentions')
        past_key_values = kwargs.get('past_key_values')
        attention_mask = kwargs.get('attention_mask')
        head_mask = kwargs.get('head_mask')

        self.batch_size = hidden_states.shape[0]
        self.output_shape = hidden_states.shape[1:]

        self.output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        if past_key_values is None:
            self.past_key_values = tuple([None] * (len(self.h)))  # same length as input past_key_values
        if attention_mask is not None:
            if self.batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(self.batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
        self.attention_mask = attention_mask
        self.use_cache = use_cache if use_cache is not None else self.config.use_cache
        self.head_mask = self.get_head_mask(head_mask, self.config.n_layer)

    def forward(self, hidden_states, lora_state, model_index, **kwargs):
        if model_index == 0:
            self.init_param(hidden_states, **kwargs)

        if hidden_states is None:
            raise ValueError("You have to specify hidden_states")
        hidden_states.to(self.h[model_index].ln_1.weight.device)

        layer_past = self.past_key_values[model_index]
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(hidden_states.device)
            # Ensure layer_past is on same device as hidden_states (might not be correct)
            if layer_past is not None:
                layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
            # Ensure that attention_mask is always on the same device as hidden_states
            if self.attention_mask is not None:
                self.attention_mask = self.attention_mask.to(hidden_states.device)
            if isinstance(self.head_mask, torch.Tensor):
                self.head_mask = self.head_mask.to(hidden_states.device)

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, self.use_cache, self.output_attentions)

                return custom_forward

            outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.h[model_index]),
                hidden_states,
                lora_state,
                None,
                self.attention_mask,
                self.head_mask[model_index],
            )
        else:
            outputs = self.h[model_index](
                hidden_states,
                lora_state=lora_state,
                layer_past=layer_past,
                attention_mask=self.attention_mask,
                head_mask=self.head_mask[model_index],
                use_cache=self.use_cache,
                output_attentions=self.output_attentions,
            )

        hidden_states = outputs[0]

        if model_index == len(self.h) - 1:
            hidden_states = self.ln_f(hidden_states)
            hidden_states.view(self.output_shape)

        # result to cpu and return
        hidden_states = hidden_states.to("cpu")
        return hidden_states


