from sklearn.base import BaseEstimator
from typing import Literal
import numpy as np
from LossFunction import MSE
import matplotlib.pyplot as plt

BATCH_NUMBER_DEFAULT = 5

class MyLinearRegression():
    def __init__(self, eta: float = 1, tol: float = 1e-6, max_iters: int = 1000, fit_intercept: bool = True, save_loss_log : bool = False) -> None:
        self.__coef: np.ndarray = None
        self.__eta: float = eta
        self.__tol: float = tol
        self.__n_iters: int = max_iters
        self.__fit_intercept: bool = fit_intercept
        if (save_loss_log):
            self.__loss_log = []
        else:
            self.__loss_log = None
    def __str__(self) -> str:
        pass
    def fit_with_BGD(self, X, y):
        self.__X: np.ndarray = X
        X_train: np.ndarray = X
        self.__y: np.ndarray = y
        n_instances = len(self.__X)

        if (self.__fit_intercept):
            X_train = np.c_[np.ones((n_instances, 1)), self.__X]

        self.__n_features: int = len(X_train[0])

        # Using Batch Gradient Descent (BGD) to optimize the loss function
        # Using the MSE (Mean Square Error) loss function

        # Initialize the non-training model's parameters
        self.__coef = np.random.randn(self.n_features, 1)
        gradients_coef = np.zeros(self.__coef.shape)
        loss = MSE(self.coef, X_train, self.__y)
        best_loss = MSE(self.coef, X_train, self.__y)

        # Iterations
        for epoch in range(self.__n_iters):
            for j in range(len(gradients_coef)):
                sum = 0
                for i in range(n_instances):
                    sum += (self.coef.T.dot(X_train[i]) -
                            self.__y[i]) * (X_train[i][j])
                gradients_coef[j] = (2 / n_instances) * sum

            # Update the coef and intercept
            self.__coef -= self.__eta * gradients_coef
            loss = MSE(self.coef, X_train, self.__y)

            # Check for tolerance break
            if (abs(loss - best_loss) < self.__tol):
                break

            # Update best loss
            if (best_loss > loss):
                best_loss = loss

            print(f"Epoch {epoch}: After Loss = {loss[0]}")

            if (self.__loss_log != None):
                self.__loss_log.append([epoch, loss[0]])

        self.__X = X_train

    def predict(self, X: np.ndarray):
        if (self.__fit_intercept):
            X_predict = np.c_[np.ones((len(X), 1)), X]
        else:
            X_predict = X
        return self.coef.T.dot(X_predict.T)

    @property
    def coef(self) -> np.ndarray:
        return self.__coef

    @property
    def n_features(self) -> int:
        return self.__n_features
    
    @property
    def loss_log(self) -> list[list[int, float]]:
        return self.__loss_log


class MySGDRegression():
    def __init__(self, t0: float = 1, t1: float = 1, tol: float = 0, max_iters: int = 100, fit_intercept: bool = True) -> None:
        self.__t0 = t0
        self.__t1 = t1
        self.__tol = tol
        self.__max_iters = max_iters
        self.__fit_intercept = fit_intercept

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.__X: np.ndarray = X
        X_train: np.ndarray = X
        self.__y: np.ndarray = y
        n_instances = len(self.__X)
        if (self.__fit_intercept):
            X_train = np.c_[np.ones((n_instances, 1)), self.__X]
        self.__n_features: int = len(X_train[0])

        # Using Stochastic Gradient Descent (SGD) to optimize the loss function
        # Using the MSE (Mean Square Error) loss function

        # Initialize the non-training model's parameters
        self.__coef = np.random.randn(self.n_features, 1)
        gradients_coef = 0
        loss = MSE(self.coef, X_train, self.__y)
        best_loss = MSE(self.coef, X_train, self.__y)

        # Iteration:
        for epoch in range(self.__max_iters):
            for i in range(n_instances):
                index = np.random.randint(0, n_instances)
                X_index = X_train[index:index + 1]
                y_index = self.__y[index:index + 1]
                gradients_coef = 2 * X_index.T.dot(self.__coef.T.dot(X_index.T) - y_index[0])
                eta = self.__learning_schedules(epoch * n_instances + i)

                # Update the coef and intercept
                self.__coef -= eta * gradients_coef
                loss = MSE(self.coef, X_train, self.__y)

                # Check for tolerance break
                if (abs(loss - best_loss) < self.__tol):
                    return

                # Update best loss
                if (best_loss > loss):
                    best_loss = loss

            print(f"Epoch {epoch}: After Loss = {loss[0]}")

    def predict(self, X: np.ndarray):
        if (self.__fit_intercept):
            X_predict = np.c_[np.ones((len(X), 1)), X]
        else:
            X_predict = X
        return self.coef.T.dot(X_predict.T)

    def __learning_schedules(self, t):
        return self.__t0 / (t + self.__t1)

    @property
    def coef(self) -> np.ndarray:
        return self.__coef

    @property
    def n_features(self):
        return self.__n_features

class MyMiniBatchRegression():
    def __init__(self, t0: float = 1, t1: float = 1, batch_size : int = -1, tol: float = 0, max_iters: int = 100, fit_intercept: bool = True) -> None:
        self.__t0 = t0
        self.__t1 = t1
        self.__tol = tol
        self.__max_iters = max_iters
        self.__fit_intercept = fit_intercept
        self.__batch_size = batch_size

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.__X: np.ndarray = X
        X_train: np.ndarray = X
        self.__y: np.ndarray = y
        n_instances = len(self.__X)
        if (self.__fit_intercept):
            X_train = np.c_[np.ones((n_instances, 1)), self.__X]
        if (self.__batch_size == -1):
            self.__batch_size = n_instances // BATCH_NUMBER_DEFAULT
        self.__n_features: int = len(X_train[0])

        # Using Stochastic Gradient Descent (SGD) to optimize the loss function
        # Using the MSE (Mean Square Error) loss function

        # Initialize the non-training model's parameters
        self.__coef = np.random.randn(self.n_features, 1)
        gradients_coef = 0
        loss = MSE(self.coef, X_train, self.__y)
        best_loss = MSE(self.coef, X_train, self.__y)

        # Iteration:
        for epoch in range(self.__max_iters):
            for i in range(n_instances):
                gradients_coef = 0
                for batch in range(self.__batch_size):
                    index = np.random.randint(0, n_instances)
                    X_index = X_train[index:index + 1]
                    y_index = self.__y[index:index + 1]
                    gradients_batch = 2 * X_index.T.dot(self.__coef.T.dot(X_index.T) - y_index[0])
                    gradients_coef += gradients_batch
                gradients_coef *= (1 / self.__batch_size)
                eta = self.__learning_schedules(epoch * n_instances + i)

                # Update the coef and intercept
                self.__coef -= eta * gradients_coef
                loss = MSE(self.coef, X_train, self.__y)

                # Check for tolerance break
                if (abs(loss - best_loss) < self.__tol):
                    return

                # Update best loss
                if (best_loss > loss):
                    best_loss = loss

            print(f"Epoch {epoch}: After Loss = {loss[0]}")

    def predict(self, X: np.ndarray):
        if (self.__fit_intercept):
            X_predict = np.c_[np.ones((len(X), 1)), X]
        else:
            X_predict = X
        return self.coef.T.dot(X_predict.T)
    
    def __learning_schedules(self, t):
        return self.__t0 / (t + self.__t1)
    
    @property
    def coef(self) -> np.ndarray:
        return self.__coef

    @property
    def n_features(self):
        return self.__n_features