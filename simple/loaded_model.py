# This is a PyTorch version of [lstm_text_generation.py](https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py)
# in keras example using GRU instead of LSTM.

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# variables
model_name = ""
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


model = torch.load("")


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
raw_text = ""
with open(file_name) as f:
    raw_text = f.read().lower()

chars = sorted(list(set(raw_text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

sentences = []
next_chars = []
for i in range(0, len(raw_text) - maxlen, step):
    sentences.append(raw_text[i: i + maxlen])
    next_chars.append(raw_text[i + maxlen])

X = np.zeros((maxlen, len(sentences)), dtype=np.int)
y = np.zeros((len(sentences)), dtype=np.int)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[t, i] = char_indices[char]
    y[i] = char_indices[next_chars[i]]
features = len(chars)

if cuda:
    model.cuda()


def test(x, hidden):
    model.eval()
    model.zero_grad()
    output, hidden = model(x, var_pair(hidden))
    return output, hidden


def main():
    # for epoch in range(0, 30):
    #     train()

    start_index = random.randint(0, len(raw_text) - maxlen - 1)
    generated = ''
    sentence = raw_text[start_index: start_index + maxlen]
    generated += sentence
    print(sentence + "\n---")

    for i in range(1500):
        hidden = model.init_hidden(1)
        x = np.zeros((maxlen, 1), dtype=np.int)
        for t, char in enumerate(sentence):
            x[t, 0] = char_indices[char]
        x = var(torch.LongTensor(x))
        pred, hidden = test(x, hidden)
        next_idx = torch.max(pred, 1)
        next_idx = int(next_idx[1].data.sum())
        next_char = indices_char[next_idx]
        generated += next_char
        sentence = sentence[1:] + next_char
        print(next_char, end="")
    print()


if __name__ == '__main__':
    main()