
import math
import copy

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss

from test_opt_client import ClientGPT2HeadModel


def lora_param_init(client_model):
    for name, param in client_model.named_parameters():
        if 'lora_A' in name:
            nn.init.kaiming_uniform_(param, a=math.sqrt(5))
        if 'lora_B' in name:
            nn.init.normal_(param, 0, 1e-5)
    return client_model


def load_sever_model():
    _path = "/bigdata/zy/FederatedScope/model/gpt2"
    _model = AutoModelForCausalLM.from_pretrained(_path)

    for param in _model.parameters():
        param.requires_grad = False

    _model.transformer.h = _model.transformer.h[1:]
    _model.transformer.wte = None
    _model.transformer.wpe = None
    return _model


def load_lora_model():
    _path = "/bigdata/zy/FederatedScope/model/gpt2"
    _model = AutoModelForCausalLM.from_pretrained(_path)

    for param in _model.parameters():
        param.requires_grad = False

    args = {'r': 8, 'lora_alpha': 32, 'lora_dropout': 0.1, "fan_in_fan_out": True}
    peft_config = LoraConfig(task_type="CAUSAL_LM", **args)
    _model = get_peft_model(_model, peft_config)

    for name, param in _model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True

    return _model


def set_client_model(client_model):
    for idx, _h in enumerate(client_model.transformer.h):
        if idx > 0:
            _h.lora_A = _h.attn.c_attn.lora_A['default']
            _h.lora_B = _h.attn.c_attn.lora_B['default']
            _h.lora_dropout = _h.attn.c_attn.lora_dropout['default']
            _h.attn, _h.ln_2, _h.mlp = None, None, None
    return client_model


def load_client_model():
    client_model = load_lora_model()
    client_model = set_client_model(client_model)
    return client_model


def load_gpt2_model():
    server_model = load_sever_model()

    client_model = load_client_model()
    # client_model = lora_param_init(client_model)

    return server_model, client_model


def model_train(model):
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500,
        num_training_steps=10000
    )
    return optimizer, scheduler


if __name__ == '__main__':
    data = torch.tensor([21106, 318, 281, 12064, 326, 8477, 257, 4876, 13, 19430,
                         257, 2882, 326, 20431, 32543, 262, 2581, 13, 198, 198,
                         21017, 46486, 25, 198, 23318, 1115, 9040, 329, 10589, 5448,
                         13, 198, 198, 21017, 18261, 25, 16, 13, 47659, 257,
                         12974, 5496, 290, 787, 1654, 284, 2291, 6088, 286, 15921,
                         290, 13701, 13, 220, 198, 17, 13, 32900, 7987, 284,
                         1394, 534, 1767, 4075, 290, 1913, 13, 220, 198, 18,
                         13, 3497, 1576, 3993, 290, 5529, 257, 6414, 3993, 7269,
                         13, 50256])
    label = torch.tensor([-100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                          -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                          -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                          -100, -100, -100, -100, -100, -100, 16, 13, 47659, 257,
                          12974, 5496, 290, 787, 1654, 284, 2291, 6088, 286, 15921,
                          290, 13701, 13, 220, 198, 17, 13, 32900, 7987, 284,
                          1394, 534, 1767, 4075, 290, 1913, 13, 220, 198, 18,
                          13, 3497, 1576, 3993, 290, 5529, 257, 6414, 3993, 7269,
                          13, 50256])
    _server_model, _client_model = load_gpt2_model()
    _model = ClientGPT2HeadModel(
        server_model=_server_model, client_model=_client_model)
    _optimizer, _scheduler = model_train(_model)
    loss_fct = CrossEntropyLoss()

    for i in range(200):
        tag_param = _client_model.transformer.h[11].lora_A.weight.data.clone()

        _optimizer.zero_grad()
        result = _model(data, label)
        print(f"loss: {result[0]}")

        loss = result[0]
        _model.backward(loss)
        _optimizer.step()
        _scheduler.step()
