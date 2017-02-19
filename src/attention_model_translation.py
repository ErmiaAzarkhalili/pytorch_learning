# this script is inspired by [practical pytorch](https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation.ipynb)
# and I do character-level seq2seq and attention
# the data is available [here](http://www.manythings.org/anki/)

import unicodedata
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

# variables
MAX_LENGTH = 15
EOS_CODE = "<EOS>"
SOS_CODE = "<SOS>"
file_path = "data/deu.txt"

# cuda setting
cuda = False
if torch.cuda.is_available():
    cuda = True


# convert tensor to variable
def variable(t):
    t = Variable(t)
    if cuda:
        t = t.cuda()
    return t


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
def map_generator(s, initial={SOS_CODE: 0, EOS_CODE: 1}):
    chars = sorted(list(set(s)))
    char_idx = dict((c, i) for i, c in enumerate(chars, len(initial)))
    char_idx.update(initial)
    idx_char = {j: i for i, j in char_idx.items()}
    return char_idx, idx_char


# read data from the text file
class DataGenerator(object):
    def var(self, list):
        # convert list to autograd.Variable
        return variable(torch.LongTensor(list).view(-1, 1))

    def __init__(self, file_path, maxlen=MAX_LENGTH):
        with open(file_path) as f:
            lines = f.read()
        lines = normalize_string(lines)
        self.char_idx, self.idx_char = map_generator(lines)
        self.lines = lines.split("\n")
        self.maxlen = maxlen

    def load(self, reverse=False, to_from=False, random_seed=None):
        """
        data iterator
        :param reverse: reverse from_text, e.g. hello<EOS> -> olleh<EOS>
        :param to_from: switch from_text and to_text
        :param random_seed: seed of random state
        :return:
        """
        indices = [i for i in range(len(self.lines))]
        random.seed(random_seed)
        random.shuffle(indices)
        for idx in indices:
            # _pair : Hi!\tCiao
            _pair = self.lines[idx]
            _from_text, _to_text = _pair.split("\t")
            if len(_from_text) < self.maxlen:
                _from = [self.char_idx[w] for w in _from_text]
                if reverse:
                    _from.reverse()
                _to = [self.char_idx[w] for w in _to_text]
                _from.append(self.char_idx[EOS_CODE])
                _to.append(self.char_idx[EOS_CODE])
                if to_from:
                    # to_lang -> from_lang
                    __from = [self.char_idx[SOS_CODE]]
                    __from.extend(_from)
                    yield self.var(_to), self.var(__from), _to_text, _from_text
                else:
                    __to = [self.char_idx[SOS_CODE]]
                    __to.extend(_to)
                    yield self.var(_from), self.var(__to), _from_text, _to_text


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.emb = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=n_layers)

    def forward(self, input, hidden):
        input = self.emb(input).view(1, 1, -1)
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def init_hidden(self):
        return variable(torch.zeros(self.n_layers, 1, self.hidden_size))


class AttentionDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1,
                 dropout_rate=0.1, max_len=MAX_LENGTH):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.max_len = max_len
        self.n_layers = n_layers

        self.emb = nn.Embedding(hidden_size, output_size)
        self.att = nn.Linear(hidden_size + output_size, max_len)
        self.att_combine = nn.Linear(hidden_size + output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=n_layers)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, e_output, e_output_seq):
        emb = self.emb(input).view(1, 1, -1)
        emb = self.dropout(emb)
        att_weights = F.softmax(self.att(torch.cat((emb[0], hidden[0]), 1)))
        att_applied = torch.bmm(att_weights.unsqueeze(0), e_output_seq.unsqueeze(0))
        output = torch.cat((emb[0], att_applied[0]), 1)
        output = self.att_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]))
        return output, hidden, att_weights

    def init_hidden(self):
        return variable(torch.zeros(self.n_layers, 1, self.hidden_size))


def _train(input, target, encoder, decoder, e_opt, d_opt,
           criterion, data_gen, max_len=MAX_LENGTH, teacher_rate=0.5):
    encoder.train()
    decoder.train()
    e_hidden = encoder.init_hidden()
    e_opt.zero_grad()
    d_opt.zero_grad()
    train_target_ratio = random.uniform(0, 1)

    i_length = input.size()[0]
    t_length = target.size()[0]
    e_output_seq = variable(torch.zeros(max_len, encoder.hidden_size))
    loss = 0

    for i in range(i_length):
        e_output, e_hidden = encoder(input[i], e_hidden)
        e_output_seq[i] = e_output[0][0]

    d_hidden = e_hidden

    if train_target_ratio < teacher_rate:
        for i in range(t_length):
            d_output, d_hidden, d_attention = decoder(target[i], d_hidden, e_output, e_output_seq)
            loss += criterion(d_output[0], target[i])
    else:
        d_output = data_gen.var([data_gen.char_idx[SOS_CODE]])
        for i in range(t_length):
            d_output, d_hidden, d_attention = decoder(d_output, d_hidden, e_output, e_output_seq)
            loss += criterion(d_output, target[i])
            d_output = torch.topk(d_output, 1)[1]

    loss.backward()
    e_opt.step()
    d_opt.step()
    return loss.data[0]/t_length


def train(encoder, decoder, n_epochs, data_gen, e_lr=1e-3, d_lr=1e-3):
    e_opt = optim.Adam(encoder.parameters(), lr=e_lr)
    d_opt = optim.Adam(decoder.parameters(), lr=d_lr)
    criterion = nn.NLLLoss()
    total_loss = 0
    loss_list = []
    print("start training")
    for epoch in range(1, n_epochs+1):
        dl = data_gen.load()
        counter = 0
        for input, target,_,_ in dl:
            loss = _train(input, target, encoder, decoder, e_opt, d_opt, criterion, data_gen)
            total_loss += loss
            loss_list.append(total_loss)
            if counter % 100 == 0:
                print("\rpartial loss{:>10.2}".format(loss), end="")
            counter += 1
        print("\nepoch {:>5},data size{:>7} pairs: loss {:>7.2}".format(epoch, counter, total_loss))
        total_loss = 0

    return loss_list


def _test(input, encoder, decoder, data_gen, limit_len, max_len=MAX_LENGTH):

    encoder.eval()
    decoder.eval()
    e_hidden = encoder.init_hidden()
    i_length = input.size()[0]
    e_output_seq = variable(torch.zeros(max_len, encoder.hidden_size))
    d_output_list = []
    for i in range(i_length):
        e_output, e_hidden = encoder(input[i], e_hidden)
        e_output_seq[i] = e_output[0][0]
    d_hidden = e_hidden
    d_output = data_gen.var([data_gen.char_idx[SOS_CODE]])
    for i in range(limit_len):
        d_output, d_hidden, d_attention = decoder(d_output, d_hidden, e_output, e_output_seq)
        d_output = torch.topk(d_output, 1)[1] # (max_val, max_idx_tensor)[1]
        _d_output = d_output.data.cpu()[0][0]
        d_output_list.append(_d_output)
        if _d_output == data_gen.char_idx[EOS_CODE]:
            break
    return d_output_list


def test(encoder, decoder, data_gen, limit_len=25):
    print("-"*10)
    dl = data_gen.load()
    input, _, question, answer = next(dl)
    prediction = _test(input, encoder, decoder, data_gen, limit_len)
    prediction = [data_gen.idx_char[i] for i in prediction]
    print("question:{} {}/{}".format(question, "".join(prediction), answer))
    print("-"*10)


def main():
    data_gen = DataGenerator(file_path)
    input_size = len(data_gen.idx_char)
    hidden_size = 128
    ecdr = Encoder(input_size, hidden_size, n_layers=1)
    dcdr = AttentionDecoder(hidden_size, input_size, n_layers=1)
    if cuda:
        ecdr.cuda()
        dcdr.cuda()
    for i in range(10):
        train(ecdr, dcdr, 1, data_gen)
        test(ecdr, dcdr, data_gen)


if __name__ == '__main__':
    main()