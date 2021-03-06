import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision.utils as vutils
from torch.nn import functional as F

# cuda setting
cuda = False
if torch.cuda.is_available():
    cuda = True

# args
parser = argparse.ArgumentParser(description="DCGAN, W(DC)GAN, LSGAN")
parser.add_argument("type", choices=["WGAN", "DCGAN", "LSGAN"], help="gan's type")
parser.add_argument("input_path")
parser.add_argument("--output_path", default="gan_output")
parser.add_argument("--n_epochs", default=200, type=int, help="num of epochs")
parser.add_argument("--bsize", default=64, type=int, help="mini batch size")
parser.add_argument("--g_lr", default=2e-4, type=float, help="learning rate for generator")
parser.add_argument("--c_lr", default=2e-4, type=float, help="learning rate for critic")
parser.add_argument("--zsize", default=32, type=int)
parser.add_argument("--load", default=False, type=bool, help="load exist model or not")
args = parser.parse_args()

# variables
batch_size = args.bsize
g_input_size = args.zsize
output_path = args.output_path

# load data
transform = transforms.Compose([
                                transforms.Scale(64),
                                transforms.CenterCrop(64),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                     (0.5, 0.5, 0.5)),
                                ])
data_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.input_path, transform=transform),
    batch_size=batch_size, shuffle=True)


def variable(t):
    t = Variable(t)
    if cuda:
        t = t.cuda()
    return t


# custom weights initialization called on Generator and Critic
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# Generator
class Generator(nn.Module):
    """(batch_size, n_z) -> (batch_size, 3, 64, 64)"""

    def __init__(self):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(g_input_size, 512, 4, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(z.size()[0], -1, 1, 1)
        return self.gen(z)


class Critic(nn.Module):
    """(batch_size, 3, 64, 64) -> ()"""

    def __init__(self):
        super(Critic, self).__init__()
        self.crtc = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        x = self.crtc(x)
        return x.view(-1, 1)


def train(generator, critic, g_opt, c_opt, gan_name, n_epochs=args.n_epochs):
    fixed_g_input = variable(torch.Tensor(64, g_input_size, 1, 1).normal_())
    one = torch.FloatTensor([1])
    g_err_list, c_err_list = [], []
    if cuda:
        one = one.cuda()

    def DC_LS(output, label, gan_name):
        if gan_name == "DCGAN":
            err = F.binary_cross_entropy(F.sigmoid(output), label)
        else:
            err = torch.norm(output-label, 2)
        return err

    def train_critic(label, g_input, data):
        critic.zero_grad()
        # train with real
        label.data.fill_(1)
        output = critic(data)
        if gan_name == "WGAN":
            c_err_real = output.mean(0).view(1)
            c_err_real.backward(one)
        else:
            c_err_real = DC_LS(output, label, gan_name)
            c_err_real.backward()
        c_x = output.data.mean()

        # train with fake
        g_input.data.normal_()
        # generator.eval()
        fake = generator(g_input)
        # generator.train()
        label.data.fill_(0)
        output = critic(fake.detach())
        if gan_name == "WGAN":
            c_err_fake = output.mean(0).view(1)
            c_err_fake.backward(-1 * one)
            c_err = c_err_real - c_err_fake
        else:
            c_err_fake = DC_LS(output, label, gan_name)
            c_err_fake.backward()
            c_err = c_err_real + c_err_fake
        c_z_1 = output.data.mean()
        c_opt.step()
        return c_err, fake, (c_x, c_z_1)

    def train_generator(label, fake):
        # update generator
        generator.zero_grad()
        label.data.fill_(1)
        # critic.eval()
        output = critic(fake)
        # critic.train()
        if gan_name == "WGAN":
            g_err = output.mean(0).view(1)
            g_err.backward(one)
        else:
            g_err = DC_LS(output, label, gan_name)
            g_err.backward()
        c_z_2 = output.data.mean()
        g_opt.step()
        return g_err, c_z_2

    for epoch in range(1, n_epochs+1):
        g_err_total, c_err_total = 0, 0
        for b_idx, (data, _) in enumerate(data_loader):
            label = variable(torch.Tensor(data.size()[0], 1))
            g_input = variable(torch.Tensor(data.size()[0], g_input_size, 1, 1))
            data = variable(data)
            c_train_num = 5 if epoch < 20 else 3

            for _ in range(c_train_num):
                c_err, fake, _ = train_critic(label, g_input, data)
            g_err, _ = train_generator(label, fake)
            g_err_total += g_err.data[0] / n_epochs
            c_err_total += c_err.data[0] / n_epochs / c_train_num

            if b_idx % 100 == 0:
                print('\r[{:>4}/{:>4}][{:>4}/{:>4}] Loss_C: {:>10.2} Loss_G: {:>10.2}'
                      .format(epoch, n_epochs, b_idx, len(data_loader),
                              c_err_total, g_err_total), end="")
        generator.eval()
        fake = generator(fixed_g_input)
        generator.train()
        vutils.save_image(fake.data,
                          '{}/fake_samples_epoch_{}.png'.format(output_path, epoch))
        g_err_list.append(g_err_total)
        c_err_list.append(c_err_total)
        print()
    return g_err_list, c_err_list


def plot(g_err, c_err):
    import matplotlib
    matplotlib.use("AGG")
    import matplotlib.pyplot as plt
    plt.plot(g_err)
    plt.plot(c_err)
    plt.legend(["g_err", "c_err"])
    plt.savefig("{}/gan_losses.png".format(output_path))


def main():
    model_path = "{}/models.pkl".format(output_path)
    if os.path.exists(model_path) and args.load:
        model = torch.load(model_path)
        generator = model["gene" \
                          "rator"]
        critic = model["critic"]
    else:
        generator = Generator()
        critic = Critic()
        generator.apply(weights_init)
        critic.apply(weights_init)
    if cuda:
        generator.cuda()
        critic.cuda()
    if args.type == "WGAN":
        g_opt = optim.RMSprop(generator.parameters(), lr=args.g_lr)
        c_opt = optim.RMSprop(generator.parameters(), lr=args.c_lr)
    else:
        g_opt = optim.Adam(generator.parameters(), lr=args.g_lr, betas=(0.5, 0.999), weight_decay=0.001)
        c_opt = optim.Adam(critic.parameters(), lr=args.c_lr, betas=(0.5, 0.999), weight_decay=0.001)
    g_err, c_err = train(generator, critic, g_opt, c_opt, args.type)
    plot(g_err, c_err)
    torch.save({"generator": generator,
                "critic": critic},
               model_path)


if __name__ == '__main__':
    main()
