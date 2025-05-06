
import math
import copy

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from encrypt_llm.llm.model.qwen2.client_model import ClientQwen2Model
from encrypt_llm.llm.model.qwen2.server_model import ServerQwen2ForCausalLM


def load_model():
    path = "/bigdata/zy/llm/model/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
    return model


def set_model_lora(model):
    for param in _model.parameters():
        param.requires_grad = False

    args = {'r': 8, 'lora_alpha': 32, 'lora_dropout': 0.1, "fan_in_fan_out": True}
    scaling = args['lora_alpha'] / args['r']
    args["target_modules"] = ["q_proj", "k_proj", "v_proj"]
    peft_config = LoraConfig(task_type="CAUSAL_LM", **args)
    model.config.scaling = scaling
    model = get_peft_model(model, peft_config)

    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
    return model


def lora_param_re_init(model):
    for name, param in model.named_parameters():
        if 'lora_A' in name:
            nn.init.kaiming_uniform_(param, a=math.sqrt(5))
        if 'lora_B' in name:
            nn.init.normal_(param, 0, 1e-5)
    return model


def set_server_model(model):
    for idx, _h in enumerate(model.model.model.layers):
        if idx > 0:
            _h.q_proj_lora_A = _h.self_attn.q_proj.lora_A['default']
            _h.q_proj_lora_B = _h.self_attn.q_proj.lora_B['default']
            _h.q_proj_lora_dropout = _h.self_attn.q_proj.lora_dropout['default']
            _h.k_proj_lora_A = _h.self_attn.k_proj.lora_A['default']
            _h.k_proj_lora_B = _h.self_attn.k_proj.lora_B['default']
            _h.k_proj_lora_dropout = _h.self_attn.k_proj.lora_dropout['default']
            _h.v_proj_lora_A = _h.self_attn.v_proj.lora_A['default']
            _h.v_proj_lora_B = _h.self_attn.v_proj.lora_B['default']
            _h.v_proj_lora_dropout = _h.self_attn.v_proj.lora_dropout['default']
            _h.self_attn, _h.mlp, _h.post_attention_layernorm = None, None, None
    return model


def set_client_model(model):
    model.model.layers = model.model.layers[1:]
    model.model.embed_tokens = None
    return model


if __name__ == '__main__':
    _model = load_model()

    _server_model = set_model_lora(copy.deepcopy(_model))
    _server_model = set_server_model(_server_model)

    _client_model = set_client_model(copy.deepcopy(_model))

    label = torch.tensor([[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                           -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                           -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                           16, 13, 47659, 257, 12974, 5496, 290, 787, 1654, 284, 2291, 6088, 286, 15921,
                           290, 13701, 13, 220, 198, 17, 13, 32900, 7987, 284, 1394, 534, 1767, 4075,
                           290, 1913, 13, 220, 198, 18, 13, 3497, 1576, 3993, 290, 5529, 257, 6414,
                           3993, 7269, 13, 50256]])

    data = torch.tensor([[21106, 318, 281, 12064, 326, 8477, 257, 4876, 13, 19430, 257, 2882,
                          326, 20431, 32543, 262, 2581, 13, 198, 198, 21017, 46486, 25, 198,
                          23318, 1115, 9040, 329, 10589, 5448, 13, 198, 198, 21017, 18261, 25,
                          16, 13, 47659, 257, 12974, 5496, 290, 787, 1654, 284, 2291, 6088, 286,
                          15921, 290, 13701, 13, 220, 198, 17, 13, 32900, 7987, 284, 1394, 534,
                          1767, 4075, 290, 1913, 13, 220, 198, 18, 13, 3497, 1576, 3993, 290, 5529,
                          257, 6414, 3993, 7269, 13, 50256]])

    _model = set_model_lora(_model)
    loss = _model(data, labels=label)
    print(loss[0])

    client_model = ClientQwen2Model(model=_client_model)
    server_model = ServerQwen2ForCausalLM(model=_server_model.model)

    hidden_states, _ = server_model(data, model_index=-1)
    for idx in range(len(_server_model.model.model.layers) - 1):
        lora_part = server_model(hidden_states, model_index=idx)
        hidden_states = client_model(hidden_states, lora_attn=lora_part, model_index=idx)
    loss = server_model.tail_forward(hidden_states, label)
    print(loss[0])
