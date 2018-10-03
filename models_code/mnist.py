from __future__ import print_function
import torch
import torch.utils.data
from torch import nn
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
from torch.nn.functional import dropout
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss


class MonteCarloDropout(torch.nn.Dropout):
    def forward(self, x):
        return torch.nn.functional.dropout(x, self.p, True, self.inplace)

class MinOfTwo(torch.nn.Module):
    """ Simple Hack for 1D min pooling. Input size = (N, C, L_in)
        Output size = (N, C, L_out) where N = Batch Size, C = No. Channels
        L_in = size of 1D channel, L_out = output size after pooling.
        This implementation does not support custom strides, padding or dialation
        Input shape compatibilty by kernel_size needs to be ensured"""
    
    def __init__(self):
        super(MinOfTwo, self).__init__()

    def forward(self, x1, x2):
        stacked = torch.stack((x1,x2))
        min_, min_indices = torch.min(stacked, dim=0)
        return min_

class Activation(torch.nn.Module):
    
    def __init__(self):
        super(Activation, self).__init__()
        self.minoftwo = MinOfTwo()
        self.threshold = torch.nn.Threshold(0,0)

    def forward(self, tensor):
        right = tensor + 1
        left = -tensor + 1

        return self.threshold(self.minoftwo(left, right))
    

class Mnist(nn.Module):
    def __init__(self):
        super(Mnist, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.dense1 = nn.Linear(400, 120)
        self.dense2 = nn.Linear(120, 84)
        self.dense3 = nn.Linear(84, 10)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        conved1 = self.maxpool(self.relu(self.conv1(x)))  # 6, 14, 14
        conved2 = self.maxpool(self.relu(self.conv2(conved1)))  # 16, 5, 5
        dense1 = self.relu(self.dense1(conved2.view(-1, 400)))
        dense2 = self.relu(self.dense2(dense1))
        dense3 = self.dense3(dense2)
        # Softmax implicit in CrossEntropyLoss

        return dense3

    def uncertainty(self, metric, predictions):
        return metric(predictions)

    def accuracy(self, target, preds):
        return accuracy_score(target, np.argmax(preds, axis=1))

    def log_loss(self, target, preds):
        return log_loss(target, preds)


class ISMnist(Mnist):
    def __init__(self):
        super(ISMnist, self).__init__()

        self.dense3 = nn.Linear(84, 10, bias=False)
        self.bar = 1
        self.act = Activation()

    def set_bar(self, bar):
        self.bar = bar

    def cauchy_activation(self, x):
        # return 1 + np.cos(x * np.pi)
        return 1 / (1 + 10 * x ** 2)
        # return torch.exp(-x**2/2)
        # return self.act(x)
        

    def forward(self, x):
        conved1 = self.maxpool(self.relu(self.conv1(x)))  # 6, 14, 14
        conved2 = self.maxpool(self.relu(self.conv2(conved1)))  # 16, 5, 5

        dense1 = self.relu(self.dense1(conved2.view(-1, 400)))
        dense2 = self.cauchy_activation(self.dense2(dense1))
        dense3 = self.dense3(dense2)
        # Softmax implicit in CrossEntropyLoss

        batch_size = conved1.shape[0]
        inhibited_channel = Variable(
            torch.ones((batch_size, 1)) * self.bar
        ).cuda()

        with_inhibited = torch.cat((dense3, inhibited_channel), dim=1)

        return with_inhibited

    def after_activation(self, x):

        conved1 = self.maxpool(self.relu(self.conv1(x)))  # 6, 14, 14
        conved2 = self.maxpool(self.relu(self.conv2(conved1)))  # 16, 5, 5

        dense1 = self.relu(self.dense1(conved2.view(-1, 400)))
        dense2 = self.cauchy_activation(self.dense2(dense1))

        return dense2


class MCMnist(Mnist):

    def __init__(self):
        super(MCMnist, self).__init__()

        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        conved1 = self.dropout(self.maxpool(self.relu(self.conv1(x))))  # 6, 14, 14
        conved2 = self.dropout(self.maxpool(self.relu(self.conv2(conved1))))  # 16, 5, 5

        dense1 = self.dropout(self.relu(self.dense1(conved2.view(-1, 400))))
        dense2 = self.dropout(self.relu(self.dense2(dense1)))
        dense3 = self.dense3(dense2)
        # Softmax implicit in CrossEntropyLoss

        return dense3


def train(
        epoch,
        model,
        train_loader,
        optimizer,
        loss_function,
        log_interval,
        num_batches,
        channels=1
):
    model.train()
    train_loss = 0
    accuracy = 0
    for batch_idx, (data, y) in enumerate(train_loader):
        data = Variable(data)
        data = data.cuda()
        optimizer.zero_grad()
        y_ = model(data.view(-1, channels, 32, 32))
        loss = loss_function(y_, Variable(y).cuda())
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()

        accuracy += accuracy_score(y, np.argmax(y_.cpu().data.numpy(), axis=1))
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f} Average accuracy: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset), accuracy / num_batches))


def test(
        epoch,
        model,
        test_loader,
        optimizer,
        loss_function,
        log_interval,
        channels=1
):
    model.eval()
    test_loss = 0
    y_s = []
    ys = []
    softmax = nn.Softmax()
    for i, (data, y) in enumerate(test_loader):
        data = data.cuda()
        data = Variable(data, volatile=True)
        y_ = model(data.view(-1, channels, 32, 32))
        y_s.append(softmax(y_).cpu().data.numpy())
        ys.append(y)

        test_loss += loss_function(y_, Variable(y).cuda()).data[0]

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

    print('Test accuracy: {}'.format(accuracy_score(
        np.concatenate(ys),
        np.argmax(np.concatenate(y_s), axis=1)
    )))


def perform_training(epochs, model, train_loader, test_loader,
                     optimizer, loss_function, log_interval, savepath,
                     num_batches, channels=1):

    for epoch in range(1, epochs + 1):
        train(
            epoch,
            model,
            train_loader,
            optimizer,
            loss_function,
            log_interval,
            num_batches,
            channels
        )
        test(
            epoch,
            model,
            test_loader,
            optimizer,
            loss_function,
            log_interval,
            channels
        )

    torch.save(model.state_dict(), savepath)


def load_data(batch_size):

    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(2),
                           transforms.ToTensor()])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False,
                       transform=transforms.Compose([
                           transforms.Pad(2),
                           transforms.ToTensor()])),
        batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader

