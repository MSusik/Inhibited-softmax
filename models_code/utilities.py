from torch import optim
import pickle
import torch


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
