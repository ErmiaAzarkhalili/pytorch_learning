import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable

# variables
seq_len = 10
batch_size = 32

def data_generator(length, batch_size, test_size=3):
    epsilon = 1e-4
    def e_cos(x):
            return np.exp(-np.cos(x))


    while True:
        train_c, test_c = 0, 0
        train_X, train_Y = np.zeros((batch_size, length)), []
        test_X, test_Y = np.zeros((test_size, length)), []
        test_ids = random.sample(range(batch_size + test_size), test_size)
        for i in range(batch_size + test_size):
            if i not in test_ids:
                tmp = np.arange(i * epsilon, (i + length) * epsilon, epsilon)
                print(len(tmp))
                train_X[train_c] = e_cos(tmp)
                train_Y.append(e_cos((i + length) * epsilon))
                train_c += 1
            else:
                tmp = np.arange(i * epsilon, (i + length) * epsilon, epsilon)
                test_X[test_c] = e_cos(tmp)
                test_Y.append(e_cos((i + length) * epsilon))
                test_c += 1

        yield train_X, np.array(train_Y), test_X, np.array(test_Y)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.rnn1 = nn.GRU(input_size=seq_len,
                            hidden_size=128,
                            num_layers=1)
        self.dense1 = nn.Linear(128, 1)

    def forward(self, x, hidden):
        x, hidden = self.rnn1(x, hidden)
        x = self.dense1(x)
        return x, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return Variable(weight.new(128, batch_size, 1).zero_())