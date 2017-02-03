import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import tweepy
import json
import time

# variables
model_name = "neko.txt.rnn.mdl"
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

print("loading model")
model = torch.load(model_name)
if cuda:
    model.cuda()

# raw_text, sentences, char_indices, indices_char, features, X, y

char_dict = torch.load(file_name+".chardict")
raw_text = char_dict["raw_text"]
sentences = char_dict["sentences"]
char_indices = char_dict["char_indices"]
indices_char = char_dict["indices_char"]
features = char_dict["features"]
X, y = char_dict["X"], char_dict["y"]


def test(x, hidden):
    model.eval()
    model.zero_grad()
    output, hidden = model(x, var_pair(hidden))
    return output, hidden


def reply(seed, name):

    if len(seed) > maxlen:
        seed = seed[len(seed)-maxlen-1:]

    start_index = random.randint(0, len(raw_text) - maxlen - 1)
    sentence = raw_text[start_index: start_index+maxlen-len(seed)] + seed
    print(len(sentence))
    reply_chars = ""
    # if len(seed) > 10:
    #     reply_chars = seed[-10:]
    # else:
    #     reply_chars = seed
    # print(sentence + "\n---")

    for i in range(140-len(name)-15):
        hidden = model.init_hidden(1)
        x = np.zeros((maxlen, 1), dtype=np.int)
        for t, char in enumerate(sentence):
            x[t, 0] = char_indices.get(char, 0)
        x = var(torch.LongTensor(x))
        pred, hidden = test(x, hidden)
        next_idx = torch.max(pred, 1)
        next_idx = int(next_idx[1].data.sum())
        next_char = indices_char.get(next_idx, "☃")
        sentence = sentence[1:] + next_char
        reply_chars += next_char
        # print(next_char, end="")
    # print()
    return reply_chars

with open("passwd.json") as f:
    passwd = json.load(f)

auth = tweepy.OAuthHandler(passwd["c_key"], passwd["c_sec"])
auth.set_access_token(passwd["a_key"], passwd["a_sec"])

api = tweepy.API(auth)

while True:
    time.sleep(60)
    timeline = api.mentions_timeline()
    with open("reply.log") as f:
        already = f.read().split(",")
    for status in timeline:
        status_id = status.id
        if str(status_id) not in already:
            status_txt = status.text.split(" ")[1]
            screen_name = status.author.screen_name
            reply_text = "@" + screen_name + " " + reply(status_txt, screen_name)
            print(reply(status_txt, screen_name))
            api.update_status(status=reply_text, in_reply_to_status_id=status_id)

            with open("reply.log", 'a') as f:
                f.write(str(status_id) + ",")


