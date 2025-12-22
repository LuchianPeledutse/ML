import sys
sys.path.append("C:\\main\\GitHub\\ML\\losses")

import numpy as np
from linear_mse import MSE
from linear_log_loss import LogLoss, sigmoid


class LinearRegression:
    """Numpy linear regression implementation"""
    def __init__(self, epoch: int = 100, alpha: float = 0.05):
        # Hyperparameters
        self.alpha = alpha
        self.epoch = epoch
        # Training attributes
        self.num_features = None
        self.losses_list = []
        self.loss = MSE()
        self.weight = None
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Given feature matrix returns a vectors of linear predictions

        Args
        ----
        X: np.ndarray
            Feature matrix of shape NxD where N-number of samples, D-number of features
        
        Returns
        -------
        prediction: np.ndarray
            Prediction vector of shape Nx1 for each object in feature matrix
        """
        assert type(self.weight) != type(None), "The weight matrix is not initialized. Prediction is possible after fitting"
        if X.shape[1] != self.num_features:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X@self.weight
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> self:
        """
        Fits linear regression given feature matrix X and target matrix y

        Args
        ----
        X: np.ndarray
            Feature matrix of shape NxD where N-number of samples, D-number of features
        y: np.ndarray
            Target column vector of shape Nx1
        
        Returns
        -------
        Fitted Linear regression object
        """
        # Add constant vector to X
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        # Initialize the weight vector
        self.num_features = X.shape[1]
        self.weight = np.random.default_rng().random((self.num_features, 1))
        # Training loop
        for _ in range(self.epoch):
            y_pred = self.predict(X)
            the_loss = self.loss(y_pred, y)
            self.losses_list.append(the_loss)
            
            grad = self.loss.grad(self.weight, X, y)
            self.weight -= self.alpha*grad
        # Fix bias
        self.bias = self.weight[0,0]
        return self


class LogisticRegression(LinearRegression):
    """Numpy logistic regression implementation"""
    def __init__(self, epoch: int = 1000, alpha: float = 0.005, threshold: float = 0.5):
        super().__init__(epoch, alpha)
        self.threshold = threshold
        self.loss = LogLoss()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Given feature matrix returns a vectors of linear logistic predictions

        Args
        ----
        X: np.ndarray
            Feature matrix of shape NxD where N-number of samples, D-number of features
        
        Returns
        -------
        prediction: np.ndarray
            Prediction vector of shape Nx1 for each object in feature matrix
        """
        assert type(self.weight) != type(None), "The weight matrix is not initialized. Prediction is possible after fitting"
        if X.shape[1] != self.num_features:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        return (sigmoid(X@self.weight) > self.threshold).astype(np.uint8)
    