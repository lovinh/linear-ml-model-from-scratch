from sklearn.base import BaseEstimator
from typing import Literal
import numpy as np
from LossFunction import MSE
import matplotlib.pyplot as plt

class MyLinearRegression(BaseEstimator):
    def __init__(self, eta: float = 1, tol: float = 1e-6, max_iters: int = 1000, fit_intercept: bool = True) -> None:
        self.__coef : np.ndarray = None
        self.__eta : float = eta
        self.__tol : float  = tol
        self.__n_iters : int = max_iters
        self.__fit_intercept : bool = fit_intercept
        super().__init__()

    def fit(self, X, y):
        self.__X : np.ndarray = X
        X_train : np.ndarray = X
        self.__y : np.ndarray = y
        n_instances = len(self.__X)

        if (self.__fit_intercept):
            X_train = np.c_[np.ones((n_instances, 1)), self.__X]
        
        self.__n_features : int = len(X_train[0])

        # Using Batch Gradient Descent (BGD) to optimize the loss function
        # Using the MSE (Mean Square Error) loss function
        
        # Initialize the non-training model's parameters
        self.__coef = np.random.randn(self.n_features, 1)
        gradients_coef = np.zeros(self.__coef.shape)
        loss = MSE(self.coef, X_train, self.__y)
        best_loss = MSE(self.coef, X_train, self.__y)

        # Iterations
        for epoch in range(self.__n_iters):
            print(f"Epoch {epoch}: Before Loss = {loss}\nCoef = {self.coef}")
            for j in range(len(gradients_coef)):
                sum = 0
                for i in range(n_instances):
                    sum += (self.coef.T.dot(X_train[i]) - self.__y[i]) * (X_train[i][j])
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

            print(f"Epoch {epoch}: After Loss = {loss}\nCoef = {self.coef}\nBest Loss = {best_loss}")

    def predict(self, X: np.ndarray):
        pass

    @property
    def coef(self) -> np.ndarray:
        return self.__coef

    @property
    def n_features(self) -> int:
        return self.__n_features