from __future__ import print_function
import torch
import torch.utils.data
from torch import nn
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms


class Cifar(nn.Module):
    def __init__(self):
        super(Cifar, self).__init__()

        self.conv1 = nn.Conv2d(3, 80, 3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(80)
        self.conv2 = nn.Conv2d(80, 160, 3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(160)
        self.conv3 = nn.Conv2d(160, 240, 3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(240)

        self.dense1 = nn.Linear(240 * 4 * 4, 200)
        self.batchnorm4 = nn.BatchNorm1d(200)
        self.dense2 = nn.Linear(200, 100)
        self.dense3 = nn.Linear(100, 10)

        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        conved1 = self.relu(
            self.batchnorm1(self.conv1(x)))  # 64, 16, 16

        maxpooled1 = self.dropout(self.maxpool(conved1))  # 128, 4, 4

        conved2 = self.relu(
            self.batchnorm2(self.conv2(maxpooled1)))  # 192, 2, 2
        maxpooled2 = self.dropout(self.maxpool(conved2))  # 64, 4, 4
        conved3 = self.relu(
            self.batchnorm3(self.conv3(maxpooled2)))  # 64, 8, 8
        maxpooled3 = self.dropout(self.maxpool(conved3))  # 64, 4, 4

        dense1 = self.dropout(self.relu(
            self.batchnorm4(self.dense1(maxpooled3.view(-1, 240 * 4 * 4)))))
        dense2 = self.relu(self.dense2(dense1))
        dense3 = self.dense3(dense2)

        return dense3


class ISCifar(Cifar):
    def __init__(self):
        super(ISCifar, self).__init__()

        self.bar = 1
        self.dense3 = nn.Linear(100, 10, bias=False)

    def cauchy_activation(self, x):
        return 1 / (1 + x ** 2)

    def forward(self, x):
        conved1 = self.relu(
            self.batchnorm1(self.conv1(x)))  # 64, 16, 16

        maxpooled1 = self.dropout(self.maxpool(conved1))  # 128, 4, 4

        conved2 = self.relu(
            self.batchnorm2(self.conv2(maxpooled1)))  # 192, 2, 2
        maxpooled2 = self.dropout(self.maxpool(conved2))  # 64, 4, 4
        conved3 = self.relu(
            self.batchnorm3(self.conv3(maxpooled2)))  # 64, 8, 8
        maxpooled3 = self.dropout(self.maxpool(conved3))  # 64, 4, 4

        dense1 = self.dropout(self.relu(
            self.batchnorm4(self.dense1(maxpooled3.view(-1, 240 * 4 * 4)))))
        dense2 = self.cauchy_activation(self.dense2(dense1))
        dense3 = self.dense3(dense2)
        # Softmax implicit in CrossEntropyLoss

        batch_size = conved1.shape[0]
        inhibited_channel = Variable(
            torch.ones((batch_size, 1)) * self.bar
        ).cuda()

        with_inhibited = torch.cat((dense3, inhibited_channel), dim=1)

        return with_inhibited, dense2


class DeVriesCifar(Cifar):

    def __init__(self):
        super(DeVriesCifar, self).__init__()

        self.confidence = nn.Linear(100, 1)

    def forward(self, x):
        conved1 = self.relu(
            self.batchnorm1(self.conv1(x)))  # 64, 16, 16

        maxpooled1 = self.dropout(self.maxpool(conved1))  # 128, 4, 4

        conved2 = self.relu(
            self.batchnorm2(self.conv2(maxpooled1)))  # 192, 2, 2
        maxpooled2 = self.dropout(self.maxpool(conved2))  # 64, 4, 4
        conved3 = self.relu(
            self.batchnorm3(self.conv3(maxpooled2)))  # 64, 8, 8
        maxpooled3 = self.dropout(self.maxpool(conved3))  # 64, 4, 4

        dense1 = self.dropout(self.relu(
            self.batchnorm4(self.dense1(maxpooled3.view(-1, 240 * 4 * 4)))))
        dense2 = self.relu(self.dense2(dense1))
        dense3 = self.dense3(dense2)
        confidence = self.confidence(dense2)

        return dense3, confidence


def load_data(batch_size, shuffle=True):

    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,
                         transform=transforms.Compose([
                           transforms.ToTensor()])),
        batch_size=batch_size, shuffle=shuffle, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False,
                         transform=transforms.Compose([
                           transforms.ToTensor()])),
        batch_size=batch_size, shuffle=shuffle, **kwargs)

    return train_loader, test_loader


def load_svhn(batch_size, shuffle=True):

    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.SVHN('../data', download=True,
                         transform=transforms.Compose([
                           transforms.ToTensor()])),
        batch_size=batch_size, shuffle=shuffle, **kwargs)

    return train_loader
