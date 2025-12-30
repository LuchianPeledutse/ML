import sys
sys.path.append(r"C:\main\GitHub\pythonAlgorithms\dataStructures")

import numpy as np

from utils_models import MSE, Criterion
from binaryTree import BinaryNode, BinaryTree




class DecisionTree(BinaryTree):
    def __init__(self, criterion: Criterion,
                 root: BinaryNode | None = None, crit_max_diff_eps: float = 0.01):
        super().__init__(root)
        self.crit_max_diff_eps = crit_max_diff_eps
        self.criterion = criterion
    
    def fit(self, X: np.ndarray, y: np.ndarray, split: str | None = None, key: int = 0,
            value: float | None = None, split_ind: int | None = None) -> self:
        """
        Fits decision tree model given feature matrix X and target vector y

        Args
        ----
        X: np.ndarray
            Feature matrix of shape NxD
        y: np.ndarray
            Target vector of shape Nx1 
        """
        if X.shape[0] == 0 or X.shape[0] == 1:
            return self
        # Keep track of statistics
        max_split_criterion = float('-inf')
        value = value
        split_ind = split_ind

        for feature_ind in range(X.shape[1]):
            for obj_ind in range(X.shape[0]):
                t_value = X[obj_ind, feature_ind].item()
                # Split the dataset by t_value to calculate loss
                y_left = y[X[:, feature_ind] <= t_value, :]
                y_right = y[X[:, feature_ind] > t_value, :]
                # Calculate the losses to find split criterion value
                y_loss = self.criterion(y)
                y_left_crit = self.criterion(y_left)
                y_right_crit = self.criterion(y_right)
                # Calculate branch criterion to decide on splitting
                split_crit = y_loss - (len(y_left)/len(y)*y_left_crit + len(y_right)/len(y)*y_right_crit)
                if split_crit > max_split_criterion:
                    max_split_criterion = split_crit
                    split_ind = feature_ind
                    value = t_value
        # Check if split is needed
        if max_split_criterion <= self.crit_max_diff_eps:
            if split == 'left':
                parent = super().search(self.root, key+1)
                left_node = BinaryNode(key = key, 
                                       data = {"t": value, "split_ind": split_ind, "target": y, "branch_value": max_split_criterion},
                                       parent = parent)
                super().tree_insert(self, left_node)
            elif split == 'right':
                parent = super().search(self.root, key-1)
                right_node = BinaryNode(key = key,
                                        data = {"t": value, "split_ind": split_ind, "target": y, "branch_value": max_split_criterion},
                                        parent = parent)
                super().tree_insert(self, right_node)
        else:
            X_left = X[X[:, split_ind] <= value, :]
            y_left = y[X[:, split_ind] <= value, :]

            X_right = X[X[:, split_ind] > value, :]
            y_right = y[X[:, split_ind] > value, :]

            self.fit(X_left, y_left, split = 'left', key = key-1, value=value, split_ind=split_ind)
            self.fit(X_right, y_right, split = 'right', key = key+1, value=value, split_ind=split_ind)
            
            return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Given an array matrix X returns vectors of predictions

        Args
        ----
        X: np.ndarray
            X features matrix of shape N x D where N-number of samples, D-number of features
        """
        predictions = []
        for obj_ind in range(X.shape[0]):
            object = X[obj_ind, :]
            current_node = self.root
            while current_node.right != None or current_node.left != None:
                if object[current_node.data["split_ind"]] < current_node.data["t"]:
                    if current_node.left == None:
                        break
                    current_node = current_node.left
                else:
                    if current_node.right == None:
                        break
                    current_node = current_node.right
            predictions.append(current_node.data["target"].mean().item())
        return np.array(predictions).reshape(-1, 1)
    

'--------------------------------------------------------------------------------'