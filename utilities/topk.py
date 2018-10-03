import numpy as np

def top_k_accuracy_score(y_true, y_pred, k=5, normalize=True):
    """Top k Accuracy classification score.
    For multiclass classification tasks, this metric returns the
    number of times that the correct class was among the top k classes
    predicted.
    Parameters
    ----------
    y_true : 1d array-like, or class indicator array / sparse matrix
        shape num_samples or [num_samples, num_classes]
        Ground truth (correct) classes.
    y_pred : array-like, shape [num_samples, num_classes]
        For each sample, each row represents the
        likelihood of each possible class.
        The number of columns must be at least as large as the set of possible
        classes.
    k : int, optional (default=5) predictions are counted as correct if
        probability of correct class is in the top k classes.
    normalize : bool, optional (default=True)
        If ``False``, return the number of top k correctly classified samples.
        Otherwise, return the fraction of top k correctly classified samples.
    Returns
    -------
    score : float
        If ``normalize == True``, return the proportion of top k correctly
        classified samples, (float), else it returns the number of top k
        correctly classified samples (int.)
        The best performance is 1 with ``normalize == True`` and the number
        of samples with ``normalize == False``.
    See also
    --------
    accuracy_score
    Notes
    -----
    If k = 1, the result will be the same as the accuracy_score (though see
    note below). If k is the same as the number of classes, this score will be
    perfect and meaningless.
    In cases where two or more classes are assigned equal likelihood, the
    result may be incorrect if one of those classes falls at the threshold, as
    one class must be chosen to be the nth class and the class chosen may not
    be the correct one.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import top_k_accuracy_score
    >>> y_pred = np.array([[0.1, 0.3, 0.4, 0.2],
    ...                    [0.4, 0.3, 0.2, 0.1],
    ...                    [0.2, 0.3, 0.4, 0.1],
    ...                    [0.8, 0.1, 0.025, 0.075]])
    >>> y_true = np.array([2, 2, 2, 1])
    >>> top_k_accuracy_score(y_true, y_pred, k=1)
    0.5
    >>> top_k_accuracy_score(y_true, y_pred, k=2)
    0.75
    >>> top_k_accuracy_score(y_true, y_pred, k=3)
    1.0
    >>> top_k_accuracy_score(y_true, y_pred, k=2, normalize=False)
    3
    """
    if len(y_true.shape) == 2:
        y_true = np.argmax(y_true, axis=1)

    num_obs, num_labels = y_pred.shape
    idx = num_labels - k - 1
    counter = 0
    argsorted = np.argsort(y_pred, axis=1)
    for i in range(num_obs):
        if y_true[i] in argsorted[i, idx+1:]:
            counter += 1
    if normalize:
        return counter / num_obs
    else:
        return counter
