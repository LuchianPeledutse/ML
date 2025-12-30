import numpy as np
from collections.abc import Callable


class BootStrap:
    pass

class MSE:
    def __call__(self, y_pred: np.ndarray, y: np.ndarray) -> float:
        """
        Given two arrays of shape Nx1 returns MSE on these two arrays
        """
        return ((y_pred-y)**2).mean().item()

class Criterion:
    def __init__(self, loss: MSE, strategy: str = 'reg'):
        self.loss = loss
        self.strategy = strategy
    
    def __call__(self, target_vector: np.ndarray) -> float:
        """
        Given target vector calculates an optimal prediction and calculates loss on that prediction

        Args
        ----
        target_vectors: np.ndarray 
            Target vector of shape Nx1 where N is number of objects
        
        Returns
        -------
        optimal_loss: float
            Loss calculated with optimal prediction
        """
        y_pred = target_vector.mean() if self.strategy == 'reg' else None
        optimal_loss = self.loss(y_pred = y_pred, y = target_vector)
        return optimal_loss



'--------------------------------------------------------------------------------'