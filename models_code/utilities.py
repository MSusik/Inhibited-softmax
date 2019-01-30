from torch import optim
import pickle
import torch
from scipy.special import gammaln


def create_model(
        class_,
        loss_function=torch.nn.CrossEntropyLoss,
        optimizer=optim.Adadelta
):
    model = class_()
    model.cuda()
    optimizer = optimizer(model.parameters())

    return model, optimizer, loss_function()


def load_model(class_, path):
    model = class_()
    model.cuda()

    model.load_state_dict(torch.load(path))

    return model


class Results():
    def __init__(self, fpr, tpr, pr, re):
        pass


def dump_results(fpr, tpr, pr, re, path):

    r = Results(fpr, tpr, pr, re)
    pickle.dump(r, open(path, 'wb'))


def loss_generalizer(loss_function):
    def generalized_loss(pred, y, global_step, local_step):
        return loss_function(pred, y)
    return generalized_loss


def KL(pred):
    beta = torch.ones(1, pred.shape[1]).cuda()
    sum_pred = torch.sum(pred, dim=1).view((-1, 1))
    sum_beta = torch.sum(beta, dim=1).view((1, 1))
    lnB = torch.lgamma(sum_pred) - torch.sum(torch.lgamma(pred), dim=1).view(
        -1, 1)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1) - torch.lgamma(sum_beta)

    dg0 = torch.digamma(sum_pred)
    dg1 = torch.digamma(pred)

    kl = torch.sum((pred - beta) * (dg1 - dg0), dim=1).view(-1,
                                                            1) + lnB + lnB_uni
    return kl


def mse_loss_(model):

    def loss(pred, y, global_step, local_step):
        S = torch.sum(pred, dim=1).view((-1, 1))
        E = pred - 1
        m = pred / S
        A = torch.sum((pred - m) ** 2, dim=1).view(-1, 1)
        B = torch.sum(pred * (S - pred) / (S * S * (S + 1)), dim=1).view(
            -1,
            1
        )

        annealing_coef = min(1, global_step / (1 + local_step))

        alp = E * (1 - pred) + 1
        C = annealing_coef * KL(alp)
        return torch.mean(A + B + C)

    return loss

