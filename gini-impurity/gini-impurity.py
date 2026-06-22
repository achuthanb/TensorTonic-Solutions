import numpy as np

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    samples = (len(y_left) + len(y_right))
    if samples == 0:
        return 0.0
    if len(y_left) == 0:
        gini_left =0.0
    else:
        gini_left = 1.0
        for c in np.unique(y_left):
            p = np.sum(np.equal(y_left, c)) / len(y_left)
            gini_left-=p**2
    if len(y_right) == 0:
        gini_right =0.0
    else:
        gini_right = 1.0
        for c in np.unique(y_right):
            p = np.sum(np.equal(y_right, c)) / len(y_right)
            gini_right-=p**2

    Nl_by_n = len(y_left)/samples
    Nr_by_n = len(y_right)/samples

    return (Nl_by_n * gini_left) + (Nr_by_n * gini_right)
    