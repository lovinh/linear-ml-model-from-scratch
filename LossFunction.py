import math
import numpy as np
def MSE(params : np.ndarray, X : np.ndarray, y : np.ndarray) -> np.ndarray:
    res : float = 0
    for i in range(len(y)):
        res += (params.T.dot(X[i]) - y[i]) ** 2
    return (1/len(y) * res)