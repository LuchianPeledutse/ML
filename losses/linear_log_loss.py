import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Applies a sigmoid function to a numpy array
    """
    return 1/(1+np.exp(-z))



class LogLoss:
    """Implementation of Log Loss and its gradient"""
    def __call__(self, y_pred: np.ndarray, y: np.ndarray) -> float:
        """
        Applies log function to two column vectors of prediction and labels

        Args
        ----
        y_pred: np.ndarray
            Prediction vector (probabilities) of shape Nx1 where N is number of samples

        y: np.ndarray
            Vector of binary labels (0, 1) of shape Nx1 where N is number of samples
        """
        return (-(y*np.log(y_pred) + (1-y)*np.log(1-y_pred)).mean()).item()
    
    def grad(self, weight: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Given X feature matrix and y target vector computes gradient w.r.t weights w

        Args
        ---------
        w: np.ndarray
            Weights of shape (D+1)x1 where D is number of features
        
        X: np.ndarray
            Feature matrix of shape Bx(D+1) where B is batch size
        
        y: np.ndarray
            Target vector of shape Bx1

        Returns
        -------
        log_loss_gradient: np.ndarray
            Gradient w.r.t. weights
        """
        y_pred = sigmoid(X@weight) # Shape Bx1
        X_booled = X*y # Broadcast shape Bx1 onto shape Bx(D+1).
        X_res = X_booled-X*y_pred# y_pred shape Bx1 is broadcast onto X shape of Bx(D+1)
        return (-X_res.mean(axis=0)).reshape(-1, 1)
        
    