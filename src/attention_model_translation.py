# this script is inspired by [practical pytorch](https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation.ipynb)
# and I do charater-level seq2seq and attention
# the data is available [here](http://www.manythings.org/anki/)

import unicodedata
import re

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F


# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


# get map char->idx, idx->char
def map_generator(s, initial={0: "SOS", 1: "EOS"}):
    chars = sorted(list(set(s)))
    char_idx = dict((c, i) for i, c in enumerate(chars, len(initial)))
    char_idx.update(initial)
    idx_char = {j:i for i,j in char_idx.items()}
    return char_idx, idx_char


# read the data from text file
def readData(lang_from, lang_to="eng", reverse=False):
    with open("data/{}.txt".format(lang_from)) as f:
        lines = f.read().strip().split("\n")