import numpy as np


def entropy(X):
    return np.sum(- X * np.log(np.clip(X, 1e-6, 1)), -1)


def expected_entropy(mc_preds):
    """
    Take a tensor mc_preds [n_mc x batch_size x n_classes] and return the
    mean entropy of the predictive distribution across the MC samples.
    """

    return np.mean(entropy(mc_preds), 0)  # batch_size


def predictive_entropy(mc_preds):
    """
    Take a tensor mc_preds [n_mc x batch_size x n_classes] and return the
    entropy of the mean predictive distribution across the MC samples.
    """
    return entropy(np.mean(mc_preds, 0))


def mutual_information(mc_preds):
    """
    Calculate the BALD (Bayesian Active Learning by Disagreement) of a model;
    the difference between the mean of the entropy and the entropy of the mean
    of the predicted distribution on the n_mc x batch_size x n_classes tensor
    mc_preds. In the paper, this is referred to simply as the MI.
    """
    BALD = predictive_entropy(mc_preds) - expected_entropy(mc_preds)
    return BALD


