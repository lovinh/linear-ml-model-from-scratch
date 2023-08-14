from LinearModel import *
from LossFunction import MSE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# Dataset

X = [[48], [52], [60], [63], [70], [40], [42], [49], [55], [52]]
y = [[170, 168, 166, 175, 185, 150, 153, 161, 169, 171]]
X = np.array(X)
y = np.array(y).T

X = 2 * np.random.rand(100, 3)
y = X.dot(np.array([[4],[-2],[10]])) + np.random.randn(100, 1)

if __name__ == "__main__":
    # model = MyLinearRegression(tol=0.001, eta=0.1, max_iters=3000, fit_intercept=False)
    # model.fit_with_BGD(X, y)
    # print(model.predict(X))
    # print(model.coef)
    model = MySGDRegression(t0 = 5, t1 = 50, tol=0.0001, max_iters=50, fit_intercept=False)
    model.fit(X, y)
    print(model.predict(X))
    print(model.coef)
    model = MyMiniBatchRegression(t0 = 5, t1 = 50, tol=0.00000000001, max_iters=50, fit_intercept=False, batch_size= -1)
    model.fit(X, y)
    print(model.predict(X))
    print(model.coef)
    pass