# This is a PyTorch version of [lstm_text_generation.py](https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py)
# in keras example using GRU instead of LSTM.

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import seq_generator

# variables
file_name = "neko.txt"
maxlen = 50
step = 3
hidden_size = 512
batch_size = 16
num_layers = 2
emb_dim = 20

# cuda setting
cuda = False
if torch.cuda.is_available():
    cuda = True

# functions


def var(x):
    x = Variable(x)
    if cuda:
        return x.cuda()
    else:
        return x


def var_pair(x):
    x_1, x_2 = Variable(x[0].data), Variable(x[1].data)
    if cuda:
        return x_1.cuda(), x_2.cuda()
    else:
        return x_1, x_2


class Net(nn.Module):
    def __init__(self, features, cls_size):
        super(Net, self).__init__()
        self.emb  = nn.Embedding(num_embeddings=features,
                                 embedding_dim=emb_dim)
        self.rnn1 = nn.LSTM(input_size=emb_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers)
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, cls_size)

    def forward(self, x, hidden):
        x = self.emb(x)
        x, hidden = self.rnn1(x, hidden)
        x = x.select(0, maxlen-1).contiguous()
        x = x.view(-1, hidden_size)
        x = F.relu(self.dense1(x))
        x = F.log_softmax(self.dense2(x))
        return x, hidden

    def init_hidden(self, batch_size=batch_size):
        weight = next(self.parameters()).data
        hidden = Variable(weight.new(num_layers, batch_size, hidden_size).zero_())
        return hidden, hidden

# ---
raw_text, sentences, char_indices, indices_char, features, X, y = \
    seq_generator.seq_generator(file_name, maxlen, step)

char_dict = {"raw_text": raw_text,
             "sentences": sentences,
             "char_indices": char_indices,
             "indices_char": indices_char,
             "features": features,
             "X": X,
             "y": y}
torch.save(char_dict, file_name + ".chardict")

print("Building the Model")
model = Net(features=features, cls_size=features)
if cuda:
    model.cuda()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=7e-4)


def train():
    model.train()
    hidden = model.init_hidden()
    for epoch in range(len(sentences) // batch_size):
        X_batch = var(torch.LongTensor(X[:, epoch*batch_size: (epoch+1)*batch_size]))
        y_batch = var(torch.LongTensor(y[epoch*batch_size: (epoch+1)*batch_size]))
        model.zero_grad()
        output, hidden = model(X_batch, var_pair(hidden))
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
    print("\r{}".format(loss.data[0]), end="")


def main():
    for epoch in range(0, 40):
        train()

if __name__ == '__main__':
    main()
    torch.save(model, file_name + ".rnn.mdl")