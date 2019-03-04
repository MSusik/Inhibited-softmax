import numpy as np
import torch

from sklearn.metrics import accuracy_score
from torch.autograd import Variable
from torch import nn


class Movie(nn.Module):
    def __init__(self):
        super(Movie, self).__init__()

        self.embedding = nn.Embedding(5045, 50)
        self.dense = nn.Linear(50, 1)

    def forward(self, x):

        embedded = self.embedding(x)
        pooled = torch.mean(embedded, dim=1)
        out = self.dense(pooled)
        return out


class MCMovie(Movie):
    def __init__(self):
        super(MCMovie, self).__init__()

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        embedded = self.embedding(x)
        pooled = torch.mean(embedded, dim=1)
        dropped  = self.dropout(pooled)
        out = self.dense(dropped)
        return out


class ISMovie(Movie):
    def __init__(self):
        super(ISMovie, self).__init__()
        self.dense = nn.Linear(50, 2, bias=False)

    def cauchy_activation(self, x):
        return 1 / (1 + x ** 2)

    def forward(self, x):

        embedded = self.embedding(x)
        pooled = torch.mean(embedded, dim=1)
        pooled = self.cauchy_activation(pooled)
        out = self.dense(pooled)

        batch_size = x.shape[0]
        inhibited_channel = Variable(
            torch.zeros((batch_size, 1))
        ).cuda()

        with_inhibited = torch.cat((out, inhibited_channel), dim=1)

        return with_inhibited


def generator_out_of_matrix(matrix_, labels_, batch_size, shuffle=True):
    arange = np.arange(0, matrix_.shape[0], 1)
    if shuffle:
        np.random.shuffle(arange)
    matrix = matrix_[arange, :]
    labels = labels_[arange]
    # print(matrix.shape[0])
    index_start = 0
    end = matrix.shape[0]
    while index_start < matrix.shape[0]:
        yield (
            torch.LongTensor(np.flip(
                matrix[index_start:min(end, index_start + batch_size)],
                axis=1
            ).astype(np.float64)),
            torch.LongTensor(
                labels[index_start:min(end, index_start + batch_size)])
        )
        index_start += batch_size


def train_sentiment(
        epoch,
        model,
        train_matrix,
        train_labels,
        optimizer,
        loss_function,
        log_interval,
        batch_size,
        channels=1,
        bce=True
):
    model.train()
    train_loss = 0
    accuracy = 0
    for batch_idx, (data, y) in enumerate(generator_out_of_matrix(train_matrix, train_labels, batch_size)):
        data = Variable(data)
        data = data.cuda()
        optimizer.zero_grad()
        y_ = model(data.view(-1, 400))

        if bce:
            # BinaryCrossEntropy
            loss = loss_function(y_, Variable(y.float().view((100, 1))).cuda())
        else:
            # CrossEntropyLoss
            loss = loss_function(y_, Variable(y).cuda())
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if bce:
            accuracy += accuracy_score(y, y_.cpu().data.numpy() > 0)
        else:
            accuracy += accuracy_score(y, np.argmax(y_[:, :2].cpu().data.numpy(), axis=1))

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), 25000,
                       100. * batch_idx / 250,
                       loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f} Average accuracy: {:.4f}'.format(
          epoch, train_loss / 25000, accuracy / 250))


def test_sentiment(
        epoch,
        model,
        test_matrix,
        test_labels,
        optimizer,
        loss_function,
        log_interval,
        channels=1,
        bce=True
):
    model.eval()
    test_loss = 0
    y_s = []
    ys = []
    softmax = nn.Softmax()
    for i, (data, y) in enumerate(generator_out_of_matrix(test_matrix, test_labels, 100)):
        data = data.cuda()
        data = Variable(data, volatile=True)
        y_ = model(data.view(-1, 400))
        y_s.append(y_.cpu().data.numpy())
        ys.append(y)

        if bce:
            test_loss += loss_function(y_, Variable(y.float().view((100, 1))).cuda()).item()
        else:
            test_loss += loss_function(y_, Variable(y).cuda()).item()

    test_loss /= 25000
    print('====> Test set loss: {:.4f}'.format(test_loss))

    if bce:
        print('Test accuracy: {}'.format(accuracy_score(
            np.concatenate(ys),
            np.concatenate(y_s) > 0)
        ))
    else:
        print('Test accuracy: {}'.format(accuracy_score(
            np.concatenate(ys),
            np.argmax(np.concatenate(y_s)[:, :2], axis=1))
        ))


def perform_training_sentiment(
        epochs,
        model,
        train_matrix,
        test_matrix,
        labels,
        optimizer,
        loss_function,
        log_interval,
        savepath,
        num_batches,
        channels=1,
        bce=True
):

    for epoch in range(1, epochs + 1):
        train_sentiment(
            epoch,
            model,
            train_matrix,
            labels,
            optimizer,
            loss_function,
            log_interval,
            num_batches,
            channels,
            bce
        )
        test_sentiment(
            epoch,
            model,
            test_matrix,
            labels,
            optimizer,
            loss_function,
            log_interval,
            channels,
            bce
        )

    torch.save(model.state_dict(), savepath)
