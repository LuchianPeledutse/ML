import numpy as np



class MSE:
    """
    Implementation of MSE loss and its gradient 
    """
    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Description
        -----------
        Computes MSE for two vectors

        Arguments
        ---------
        x: np.ndarray
            Numpy array of shape Nx1 representing a vector

        y: np.ndarray
            Numpy array of shape Nx1 representing a vector
        
        Returns
        -------
        result: float
            mean squared error on provided vectors
        """
        N = len(x)
        summed_squared_error = ((x-y)**2).sum()
        mse = (summed_squared_error/N).item()
        return mse
    
    def grad(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Description
        -----------
        Given X feature matrix and y target vector computes gradient w.r.t weights w

        Arguments
        ---------
        w: np.ndarray
            Weights of shape (D+1)x1 where D is number of features
        
        X: np.ndarray
            Feature matrix of shape Bx(D+1) where B is batch size
        
        y: np.ndarray
            Target vector of shape Bx1

        Returns
        -------
        mse_gradient: np.ndarray
            Gradient w.r.t. weights
        """
        B = len(X)
        mse_gradient = 2/B * X.T@(X@w-y)
        return mse_gradient
    

