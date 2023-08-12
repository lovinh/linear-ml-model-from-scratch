from LinearModel import MyLinearRegression, MySGDRegression
from LossFunction import MSE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# Dataset

# X = [[48, 2], [52,3], [60,4], [63,5], [70,6], [40,7], [42,8], [49,9], [55,10], [52,11]]
# y = [[170, 168, 166, 175, 185, 150, 153, 161, 169, 171]]
# X = np.array(X).T
# y = np.array(y).T

# print(X)
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
# y = scaler.fit_transform(y)

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

if __name__ == "__main__":
    model = MyLinearRegression(tol=0.001, eta=0.1, max_iters=10000000, fit_intercept=True)
    model.fit_with_BGD(X, y)
    print(model.predict(X))
    print(model.coef)
    # model = MySGDRegression(t0 = 5, t1 = 500, tol=0.001, max_iters=100, fit_intercept=True)
    # model.fit(X, y)
    # print(model.predict(X))
    # print(model.coef)