# this script is inspired by [practical pytorch](https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation.ipynb)
# and I do character-level seq2seq and attention
# the data is available [here](http://www.manythings.org/anki/)

import unicodedata
import re

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

MAX_LENGTH = 10


# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?\n\t]+", r" ", s)
    return s


# get map char->idx, idx->char
def map_generator(s, initial={"SOS": 0, "EOS": 1}):
    chars = sorted(list(set(s)))
    char_idx = dict((c, i) for i, c in enumerate(chars, len(initial)))
    char_idx.update(initial)
    idx_char = {j:i for i,j in char_idx.items()}
    return char_idx, idx_char


# read the data from text file
def read_data(lang, reverse=False):
    with open("data/{}.txt".format(lang)) as f:
        lines = f.read()
    lines = normalize_string(lines)
    char_idx, idx_char = map_generator(lines)
    lines = lines.split("\n")
    for _pair in lines:
        # _pair : Hi!\tCiao
        _from, _to = _pair.split("\t")
        _from = [char_idx[w] for w in _from]
        if reverse:
            _from.reverse()
        _to = [char_idx[w] for w in _to]
        yield _from, _to


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size

        self.emb = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size,num_layers=n_layers)

    def forward(self, input, hidden):
        input = self.emb(input).view(1,1,-1)
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1,1,self.hidden_size))


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.emb = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output = self.emb(input).view(1,1,-1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros((1,1,self.hidden_size)))


class AttentionDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers,
                 dropout_rate=0.1, max_len=MAX_LENGTH):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.max_len = max_len

        self.emb = nn.Embedding(hidden_size, output_size)
        self.att = nn.Linear(hidden_size*2, max_len)
        self.atc = nn.Linear(hidden_size*2, hidden_size)
        self.drp = nn.Dropout(dropout_rate)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_output, encoder_outputs):
        emb = self.emb(input).view(1, 1, -1)
        emb = self.drp(emb)

        atweights = F.softmax(self.att(torch.cat((emb[0], hidden[0]), 1)))
        atapplied = torch.bmm(atweights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        output = torch.cat((emb[0], atapplied[0]), 1)
        output = self.atc(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]))
        return output, hidden, atweights

    def init_hidden(self):
        return Variable(torch.zeros(1, 1, self.hidden_size))




