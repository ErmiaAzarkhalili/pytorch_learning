import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import json

# variables
model_name = ""
file_name = "neko.txt"
maxlen = 50
step = 3
hidden_size = 256
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

raw_text, char_indices, indices_char, features = seq_generator(file_name)

print("loading model")
model = torch.load(model_name)


def test(x, hidden):
    model.eval()
    model.zero_grad()
    output, hidden = model(x, var_pair(hidden))
    return output, hidden


def main():
    # for epoch in range(0, 30):
    #     train()

    for epoch in range(1, 60):

        print("\n---\nepoch: {}".format(epoch))

        start_index = random.randint(0, len(raw_text) - maxlen - 1)
        generated = ''
        sentence = raw_text[start_index: start_index + maxlen]
        generated += sentence
        print(sentence + "\n---")

        for i in range(400):
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