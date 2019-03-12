from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import torch
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
import PIL


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))

    return e_x / e_x.sum()


def softmax2d(x):
    """Compute softmax values for each sets of scores in x (x is 2d array)."""
    e_x = np.exp(x - np.max(x, axis=1).reshape(-1,1))

    result = e_x / e_x.sum(axis=1).reshape(-1,1)

    if np.isnan(result).any():
        print(e_x.shape)
        raise ValueError
    return result


def softmax2d_ensemble(x, ensemble_size=5):
    """Compute softmax values for each sets of scores in x (x is 3d array)."""
    e_x = np.exp(x - np.max(x, axis=2).reshape(ensemble_size,-1,1))

    result = e_x / e_x.sum(axis=2).reshape(ensemble_size,-1,1)

    if np.isnan(result).any():
        print(e_x.shape)
        raise ValueError
    return result


def correlation_test_error_uncertainty(
        predictive_entropy,
        test_probs,
        test_target,
        num_classes=10
):
    test_entropies = predictive_entropy(test_probs)
    test_classes_predicted = np.argmax(test_probs[:,:num_classes], axis=1)
    exp3target = test_classes_predicted != test_target

    roc = roc_auc_score(exp3target, test_entropies)
    ap = average_precision_score(exp3target, test_entropies)
    fpr, tpr, _ = roc_curve(exp3target, test_entropies)
    pr, re, _ = precision_recall_curve(exp3target, test_entropies)

    return roc, ap, fpr, tpr, pr, re


def correlation_test_error_uncertainty_variational(
        predictive_entropy,
        test_probs,
        test_target
):
    test_entropies = predictive_entropy(test_probs)
    test_classes_predicted = np.argmax(np.mean(test_probs, axis=0), axis=1)
    exp3target = test_classes_predicted != test_target

    roc = roc_auc_score(exp3target, test_entropies)
    ap = average_precision_score(exp3target, test_entropies)
    fpr, tpr, _ = roc_curve(exp3target, test_entropies)
    pr, re, _ = precision_recall_curve(exp3target, test_entropies)

    return roc, ap, fpr, tpr, pr, re


def load_notmnist(batch_size):
    kwargs = {'num_workers': 1, 'pin_memory': True}
    not_mnist_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            './notMNIST_small/',
            transform=transforms.Compose([
                           transforms.Pad(2),
                           transforms.Grayscale(),
                           transforms.ToTensor()])),
        batch_size=batch_size, shuffle=True, **kwargs)

    return not_mnist_loader


def load_omniglot(batch_size):
    kwargs = {'num_workers': 1, 'pin_memory': True}
    not_mnist_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            './omniglot/',
            transform=transforms.Compose([
                           transforms.Resize((32,32)),
                           transforms.Grayscale(),
                           transforms.Lambda(lambda x: PIL.ImageOps.invert(x)),
                           transforms.ToTensor()])),
        batch_size=batch_size, shuffle=True, **kwargs)

    return not_mnist_loader


def load_cifar_bw(batch_size):

    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,
                         transform=transforms.Compose([
                           transforms.Grayscale(),
                           transforms.ToTensor()
                         ])),
        batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader


def load_lfw(batch_size):

    kwargs = {'num_workers': 2, 'pin_memory': True}
    lfw_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            './lfw-a/lfw/',
            transforms.Compose([
                transforms.Resize((32,32)),
                transforms.ToTensor(),
            ])),
        batch_size=batch_size, shuffle=False, **kwargs)

    return lfw_loader


def random_generator(batch_size, channels=1):
    i = 0
    while i < 10:
        torch.random.manual_seed(i)
        yield torch.normal(
            torch.zeros((batch_size, 32, 32, channels)),
            torch.ones((batch_size, 32, 32, channels))
        ), torch.zeros(batch_size)


def not_mnist_predictions(models, not_mnist_loader, softmaxed=True):
    for model in models:
        model.eval()
    y_truth = []
    probs = []
    images = []
    softmax = torch.nn.Softmax()

    for i, (data, y) in enumerate(not_mnist_loader):
        images.append(data.cpu().numpy())
        y_s = []
        data = data.cuda()
        data = Variable(data)

        for model_ in models:
            output_, q = model_(data[:, 0, :, :].view(-1, 1, 32, 32))
            if softmaxed:
                y_ = softmax(output_)
            else:
                y_ = output_
            y_s.append(y_.cpu().data.numpy())
        y_truth.append(y.cpu().numpy())
        probs.append(np.stack(y_s))

    return (
        np.concatenate(y_truth),
        np.concatenate(probs, axis=1),
        images
    )


def not_mnist_prediction_variational(
        model,
        not_mnist_loader,
        num_inferences,
        channels=1
):
    model.eval()
    y_truth = []
    all_probs = []
    images = []
    softmax = torch.nn.Softmax()

    for i, (data, y) in enumerate(not_mnist_loader):

        images.append(data.cpu().numpy())
        data = data.cuda()
        data = Variable(data)
        y_s = []
        for j in range(num_inferences):
            output_ = model(data[:, 0, :, :].view(-1, channels, 32, 32))
            y_ = softmax(output_)
            y_s.append(y_.cpu().data.numpy())
        y_truth.append(y.cpu().numpy())
        all_probs.append(np.stack(y_s))

    return (
        np.concatenate(y_truth),
        np.concatenate(all_probs, axis=1),
        images
    )


def prediction_variational(
        model,
        svhn_loader,
        num_inferences,
        channels=3,
        size_factor=1,
        return_images=True
):
    model.eval()
    y_truth = []
    all_probs = []
    images = []
    softmax = torch.nn.Softmax()

    for i, (data, y) in enumerate(svhn_loader):

        if return_images:
            images.append(data.cpu().numpy())
        data = data.cuda()
        data = Variable(data)
        y_s = []
        for j in range(num_inferences):
            output_ = model(data[:, :, :, :].view(
                -1,
                channels,
                32*size_factor,
                32*size_factor
            ))
            y_ = softmax(output_)
            y_s.append(y_.cpu().data.numpy())
        y_truth.append(y.cpu().numpy())
        all_probs.append(np.stack(y_s))

    return (
        np.concatenate(y_truth),
        np.concatenate(all_probs, axis=1),
        images
    )


def non_distribution(test_probs, test_entropies, other_entropies, num_all, num_test):
    disting = np.concatenate([test_entropies, other_entropies])
    target = np.zeros(num_all, dtype=np.int)
    target[num_test:] = 1

    roc = roc_auc_score(target, disting)
    ap = average_precision_score(target, disting)
    fpr, tpr, _ = roc_curve(target, disting)
    pr, re, _ = precision_recall_curve(target, disting)

    return roc, ap, fpr, tpr, pr, re


def sigmoid(x):
    return 1/(1+np.exp(-x))


def test_eval(
        model,
        test_loader,
        channels=1,
        num_classes=10,
        sentiment=False,
        is_sentiment=False,
        size_factor=1
):
    model.eval()
    all_results = []
    groundtruth = []
    probs = []
    for i, (data, y) in enumerate(test_loader):
        data = data.cuda()
        data = Variable(data)
        if not sentiment and not is_sentiment:
            y_, sec_ = model(data.view(-1, channels, 32*size_factor, 32*size_factor))
        else:
            y_, sec_ = model(data.view(-1, 400))
        all_results.append(
            y_.cpu().data.numpy()[:, :num_classes].argmax(axis=1)
        )
        probs.append(y_.cpu().data.numpy())
        groundtruth.append(y.cpu().numpy())

    if sentiment:
        return (
            sigmoid(np.concatenate(probs)),
            np.concatenate(groundtruth),
            np.concatenate(probs)
        )

    return (
        np.concatenate(all_results),
        np.concatenate(groundtruth),
        np.concatenate(probs)
    )


def test_eval_variational(
        model,
        test_loader,
        num_inferences,
        channels=1,
        sentiment=False,
        size_factor=1
):
    model.eval()
    groundtruth = []
    all_probs = []
    for i, (data, y) in enumerate(test_loader):

        data = data.cuda()
        data = Variable(data)
        probs = []
        for j in range(num_inferences):
            if sentiment:
                y_ = model(data.view(-1, 400))
                probs.append(torch.sigmoid(y_).cpu().data.numpy())
            else:
                y_ = model(data.view(-1, channels, 32*size_factor, 32*size_factor))
                probs.append(softmax2d(y_.cpu().data.numpy()))
        groundtruth.append(y.cpu().numpy())

        all_probs.append(np.stack(probs))

    return (
        np.concatenate(groundtruth),
        np.concatenate(all_probs, axis=1)
    )
