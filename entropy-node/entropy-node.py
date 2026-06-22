import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    if np.array(y).shape[0] == 0:
        return 0.0
    # Write code here
    H = 0.0
    for C in np.unique(y):
        p = np.sum(np.equal(y,C)) / (np.shape(y)[0] + 1e-15)
        H+= -p* (np.log2(p))
    return H