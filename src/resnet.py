import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# cuda setting(s)
cuda = False
if torch.cuda.is_available():
    cuda = True

# variables
batch_size = 128
epochs = 100

# load data
transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ])
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('data/cifar10', train=True, download=True,
                   transform=transform),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('data/cifar10', train=False, transform=transform),
    batch_size=batch_size, shuffle=True)


class ResBase(nn.Module):
    # basic element of res-net
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0):
        super(ResBase, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class ResBlock(nn.Module):
    # basic block of res-net
    def __init__(self, nb_filters, right=False):
        super(ResBlock, self).__init__()
        if len(nb_filters) != 4:
            raise Exception("nb_filters size must be 3")
        self.rb1 = ResBase(nb_filters[0], nb_filters[1])
        self.rb2 = ResBase(nb_filters[1], nb_filters[2], kernel_size=3, padding=1)
        self.rb3 = ResBase(nb_filters[2], nb_filters[3])
        self.rbr = ResBase(nb_filters[0], nb_filters[3])
        self.right = right

    def forward(self, x):
        # right: base or Nothing
        # left: relu(base) -> relu(base) -> base
        y = F.relu(self.rb1(x))
        y = F.relu(self.rb2(y))
        y = F.relu(self.rb3(y))
        if self.right:
            x = self.rbr(x)
        return F.relu(x+y)


class Net(nn.Module):
    # the network of res-net
    def __init__(self):
        super(Net, self).__init__()
        self.conv_1 = nn.Conv2d(3, 32, 4, stride=2)
        self.bn_1 = nn.BatchNorm2d(32)
        self.res_stack = nn.Sequential(
            ResBlock([32,32,32,128], True),
            ResBlock([128,32,32,128], False),
            ResBlock([128,32,32,128], False),
            ResBlock([128,64,64,256], True),
            ResBlock([256,64,64,256], False),
            ResBlock([256,128,128,256], False),
            ResBlock([256,128,128,512], True),
            ResBlock([512,128,128,512], False),
            ResBlock([512,256,256,512], False))
        self.avp_1 = nn.AvgPool2d(4, ceil_mode=True)
        self.dense = nn.Linear(2*2*512, 10)

    def forward(self, x):
        x = self.conv_1(x)
        x = F.relu(self.bn_1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.res_stack(x)
        x = self.avp_1(x)
        x = self.dense(x.view(-1, 2*2*512))
        x = F.log_softmax(x)
        return x

model = Net()
if cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad() # reset reset optimizer
        output = model(data)
        loss = F.nll_loss(output, target) # negative log likelihood loss
        loss.backward() # backprop
        optimizer.step()
        if batch_idx % 100 == 0:
            print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100 * batch_idx / len(train_loader), loss.data[0]), end='')


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(test_loader)  # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100 * correct / len(test_loader.dataset)))
    return test_loss

loss = []
for i in range(1, epochs + 1):
    train(i)
    loss.append(test(i))

