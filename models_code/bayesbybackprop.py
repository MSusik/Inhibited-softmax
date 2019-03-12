from __future__ import print_function
import torch
import torch.utils.data
from torch import nn
from torch.autograd import Variable
import numpy as np
import math
from sklearn.metrics import accuracy_score


class MLPBBPLayer(nn.Module):
    def __init__(self, n_input, n_output, init_variance=-1.):
        super(MLPBBPLayer, self).__init__()
        # This network is vefry sensitive to initialization of the variance
        # For Mnist we had to init the variance to -1.7, for CIFAR to -1
        self.W_mu = nn.Parameter(
            torch.Tensor(n_input, n_output).normal_(0, 0.1))
        self.W_rho = nn.Parameter(
            torch.Tensor(n_input, n_output).zero_() + init_variance
        )
        self.b_mu = nn.Parameter(torch.Tensor(n_output).normal_(0, 0.1))
        self.b_rho = nn.Parameter(
            torch.Tensor(n_output).zero_() + init_variance
        )

        self.n_input = n_input
        self.n_output = n_output

        self.softplus = torch.nn.Softplus()

    def forward(self, X):

        epsilon_W, epsilon_b = self.get_random(1)

        # compute softplus for variance
        self.W_rho_s = self.softplus(self.W_rho)
        self.b_rho_s = self.softplus(self.b_rho)

        self.W = self.transform_gaussian_samples(self.W_mu, self.W_rho_s,
                                                 epsilon_W)
        self.b = self.transform_gaussian_samples(self.b_mu, self.b_rho_s,
                                                 epsilon_b)

        self.output = torch.mm(X, self.W) + self.b.expand(X.size()[0],
                                                          self.n_output)

        return self.output

    def get_random(self, variance):
        return (
            Variable(torch.Tensor(self.n_input, self.n_output).normal_(0,
                                                                       variance).cuda()),
            Variable(torch.Tensor(self.n_output).normal_(0, variance).cuda())
        )

    def transform_gaussian_samples(self, mu, rho_after_softplus, epsilon):
        return mu + rho_after_softplus * epsilon


class BBPMnist(nn.Module):
    def __init__(self):
        super(BBPMnist, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.dense1 = MLPBBPLayer(400, 120, -1.7)
        self.dense2 = MLPBBPLayer(120, 84, -1.7)
        self.dense3 = MLPBBPLayer(84, 10, -1.7)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.softmax = nn.Softmax()

    def forward(self, x):
        conved1 = self.maxpool(self.relu(self.conv1(x)))  # 6, 14, 14
        conved2 = self.maxpool(self.relu(self.conv2(conved1)))  # 16, 5, 5
        dense1 = self.relu(self.dense1(conved2.view(-1, 400)))
        dense2 = self.relu(self.dense2(dense1))
        dense3 = self.dense3(dense2)
        # Softmax implicit in CrossEntropyLoss

        return dense3


class BBPCifar(nn.Module):
    def __init__(self):
        super(BBPCifar, self).__init__()

        self.conv1 = nn.Conv2d(3, 80, 3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(80)
        self.conv2 = nn.Conv2d(80, 160, 3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(160)
        self.conv3 = nn.Conv2d(160, 240, 3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(240)

        self.dense1 = MLPBBPLayer(240 * 4 * 4, 200, -1)
        self.batchnorm4 = nn.BatchNorm1d(200)
        self.dense2 = MLPBBPLayer(200, 100, -1)
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


class BBPMovie(nn.Module):
    def __init__(self):
        super(BBPMovie, self).__init__()

        self.embedding = nn.Embedding(5045, 50)
        self.dense = MLPBBPLayer(50, 1, -1.2)

    def forward(self, x):
        embedded = self.embedding(x)
        pooled = torch.mean(embedded, dim=1)
        # print(pooled.shape)
        out = self.dense(pooled)
        return out


def gaussian(x, mu, sigma):
    # This probably can be done easier
    scaling = 1.0 / math.sqrt(2.0 * np.pi * (sigma ** 2))
    bell = torch.exp(- (x - mu) ** 2 / (2.0 * sigma ** 2))

    return scaling * bell


def scale_mixture_prior(x, sigma_p1, sigma_p2, pi):

    first_gaussian = pi * gaussian(x, 0., sigma_p1)
    second_gaussian = (1 - pi) * gaussian(x, 0., sigma_p2)

    tensor = first_gaussian + second_gaussian

    return torch.log(tensor)


# For variational posterior
def log_gaussian(x, mu, sigma):

    return (
        - 0.5 * math.log(2 * np.pi)
        - torch.log(torch.abs(sigma))
        - (x - mu)**2 / (2 * sigma**2)
    )


def size_aware_mean(t1, t2):
    t1_sum = torch.sum(t1)
    t1_elems = t1.view(-1).shape[0]
    t2_sum = torch.sum(t2)
    t2_elems = t2.view(-1).shape[0]

    return (t1_sum + t2_sum) / (t1_elems + t2_elems)


def combined_loss(
        model,
        output,
        label,
        scale_mixture_prior,
        log_likelihood_function,
        current_batch,
        batch_size,
        epoch,
        sigma_p1=None,
        sigma_p2=None,
        pi=None
):
    num_batches = 60000 / batch_size

    # Calculate data likelihood
    log_likelihood_sum = log_likelihood_function(output, label)

    # Calculate prior

    log_prior_sum = size_aware_mean(
        scale_mixture_prior(model.dense1.W, sigma_p1, sigma_p2, pi),
        scale_mixture_prior(model.dense1.b, sigma_p1, sigma_p2, pi)
    ) + size_aware_mean(
        scale_mixture_prior(model.dense2.W, sigma_p1, sigma_p2, pi),
        scale_mixture_prior(model.dense2.b, sigma_p1, sigma_p2, pi)
    )

    # Calculate variational posterior
    log_var_posterior_sum = size_aware_mean(
        log_gaussian(model.dense1.W, model.dense1.W_mu, model.dense1.W_rho_s),
        log_gaussian(model.dense1.b, model.dense1.b_mu, model.dense1.b_rho_s),
    ) + size_aware_mean(
        log_gaussian(model.dense2.W, model.dense2.W_mu, model.dense2.W_rho_s),
        log_gaussian(model.dense2.b, model.dense2.b_mu, model.dense2.b_rho_s)
    )

    # Calculate total loss
    if epoch == 0:
        return (1 / (2 ** current_batch)) * (
        log_var_posterior_sum - log_prior_sum) + log_likelihood_sum

    # No pi scheme in case of later epochs - not mentioned in paper, but otherwise does not make much sense
    return (1 / (num_batches)) * (
    log_var_posterior_sum - log_prior_sum) + log_likelihood_sum


def train_bbp(model, optimizer, train_loader, crossentropy, batch_size, log_interval,
              num_batches, epoch, channels=1):
    model.train()
    train_loss = 0
    accuracy=0
    for batch_idx, (data, y) in enumerate(train_loader):
        data = Variable(data)
        data = data.cuda()
        optimizer.zero_grad()
        y_ = model(data.view(-1, channels, 32, 32))

        # calculate the loss
        loss = combined_loss(
            model,
            y_,
            Variable(y).cuda(),
            scale_mixture_prior,
            crossentropy,
            batch_idx + 1,
            batch_size,
            epoch,
            sigma_p1=0.5,
            sigma_p2=0.0078125,
            pi=0.5
        )

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        accuracy += accuracy_score(y, np.argmax(y_.cpu().data.numpy(), axis=1))
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f} Average accuracy: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset), accuracy / num_batches))


def test_bbp(model, test_loader, crossentropy, batch_size, epoch, channels=1):
    model.eval()
    test_loss = 0
    y_s = []
    ys = []
    for i, (data, y) in enumerate(test_loader):
        data = data.cuda()
        data = Variable(data)
        y_ = model(data.view(-1, channels, 32, 32))
        y_s.append(y_.cpu().data.numpy())
        ys.append(y)
        test_loss += combined_loss(
            model,
            y_,
            Variable(y).cuda(),
            scale_mixture_prior,
            crossentropy,
            128,
            batch_size,
            epoch,
            sigma_p1=0.5,
            sigma_p2=0.0078125,
            pi=0.5
        )
    test_loss /= len(test_loader.dataset)
    # print(test_loss)
    print(
        '====> Test set loss: {:.4f}'.format(test_loss.cpu().data.numpy()[0]))

    print('Test accuracy: {}'.format(accuracy_score(
        np.concatenate(ys),
        np.argmax(np.concatenate(y_s), axis=1)
    )))

