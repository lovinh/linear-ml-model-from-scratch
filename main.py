from LinearModel import MyLinearRegression
from LossFunction import MSE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# Dataset

X = [[48, 52, 60, 63, 70, 40, 42, 49, 55, 52]]
y = [[170, 168, 166, 175, 185, 150, 153, 161, 169, 171]]
X = np.array(X).T
y = np.array(y).T

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

if __name__ == "__main__":
    model = MyLinearRegression(tol=0.000001, eta=0.1, max_iters=1000)
    model.fit(X, y)