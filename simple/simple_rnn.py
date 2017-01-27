import pandas as pd
import numpy as np
import math
import random

import torch
import torch.nn as nn
from torch.autograd import Variable

# variables
features = 1
seq_len = 10
hidden_size = 128
batch_size = 32

def data_generator():
    pass


def var(array):
    array = torch.Tensor(array)
    return Variable(array).cuda()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.rnn1 = nn.GRU(input_size=features,
                            hidden_size=hidden_size,
                            num_layers=1)
        self.dense1 = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden):
        x, hidden = self.rnn1(x, hidden)
        x = x.select(0, seq_len-1).contiguous()
        x = x.view(-1, hidden_size)
        x = self.dense1(x)
        return x, hidden

    def init_hidden(self):
        weight = next(self.parameters()).data
        return Variable(weight.new(1, batch_size, hidden_size).zero_())


(X_train, y_train), (X_test, y_test) = data_generator()

model = Net()
model.cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())


def train():
    model.train()
    total_loss = 0
    hidden = model.init_hidden()
    for epoch in range(len(X_train)//batch_size):
        X = var(X_train[epoch*batch_size: (epoch+1)*batch_size]
                        .reshape(seq_len, batch_size, features))
        y = var(y_train[epoch*batch_size: (epoch+1)*batch_size]
                        .reshape(-1, batch_size, features))
        model.zero_grad()
        output, hidden = model(X, Variable(hidden.data))
        loss = criterion(output, y)
        total_loss += loss.data
        loss.backward()
        optimizer.step()