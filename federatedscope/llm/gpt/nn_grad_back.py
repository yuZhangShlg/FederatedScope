
import copy

import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

INPUT_GRADIENT = []


def get_input_gradient(grad):
    global INPUT_GRADIENT
    INPUT_GRADIENT = grad


class TestNNOne(nn.Module):
    def __init__(self, input_dim):
        super(TestNNOne, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return x


class TestNNTwo(nn.Module):
    def __init__(self, output_dim):
        super(TestNNTwo, self).__init__()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


def model_train(model):
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500,
        num_training_steps=10000
    )
    return optimizer, scheduler


class TwoStepTrain(nn.Module):
    def __init__(self, first_model, second_model):
        super(TwoStepTrain, self).__init__()
        self.first_model = first_model
        self.second_model = second_model

        self.x_cache = None

    def forward(self, x):
        x = self.first_model(x)
        self.x_cache = x
        x = x.detach().cpu().numpy()

        x = torch.from_numpy(x)
        x.requires_grad_(True)
        x.register_hook(get_input_gradient)
        x = self.second_model(x)
        return x

    def backward(self, loss):
        loss.backward(retain_graph=True)
        get_gradient = INPUT_GRADIENT
        torch.autograd.backward(self.x_cache, get_gradient, retain_graph=True)


class TwoStepTrainTest(nn.Module):
    def __init__(self, first_model, second_model):
        super(TwoStepTrainTest, self).__init__()
        self.first_model = first_model
        self.second_model = second_model

    def forward(self, x):
        x = self.first_model(x)
        x = self.second_model(x)
        return x


if __name__ == '__main__':
    _input_dim = 10
    _output_dim = 2

    data = torch.randint(1, 10, (_output_dim, _input_dim)).float()
    label = torch.randint(0, 2, (_output_dim, _output_dim)).float()

    _first_model = TestNNOne(_input_dim)
    _second_model = TestNNTwo(_output_dim)
    flag_weight = copy.deepcopy(_second_model.fc3.weight.data)
    loss_fct = CrossEntropyLoss()

    _model = TwoStepTrain(_first_model, _second_model)
    _optimizer, _scheduler = model_train(_model)

    for i in range(10):
        _optimizer.zero_grad()
        output = _model(data)
        _loss = loss_fct(output, label)
        _model.backward(_loss)
        _optimizer.step()
        _scheduler.step()
        # print(torch.all(flag_weight.eq(_second_model.fc3.weight.data)))
        print(_loss.cpu().detach().numpy())

