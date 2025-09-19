#imports 
import numpy as np
from tqdm import tqdm
from numpy import ndarray


#data node structure
class TreeNode(object):
    def __init__(self,data = None):
        self.data = data
        self.left = None
        self.right = None




#models
class DecisionTreeClassifier:
    """
    implements a decision tree classifier
    """
    def __init__(self, max_depth = float('inf')):
        #Hyper parameters
        self.max_depth = max_depth
        #Fitting parameters
        self.number_of_classes = None
        self.samples_right = None
        self.samples_left = None
        self.gini_right = None
        self.gini_left = None
        #The BinaryTree
        self.BinaryTree = None

    def fit(self, X_data:ndarray , y_data:ndarray):
        """
        Fits the Decision Tree
        X has shape of N x M where M is number of features; N is number of instances 
        """
        #defining self parameters during training
        self.unique_classes = np.unqiue(y_data)
        #parameters for further training
        number_of_classes = X_data.shape[1]
        number_of_instances = X_data.shape[0]
        best_row = None
        best_class = None
        #Training algorithm
        if number_of_instances == 1:
            return self
        else:
            pass

    def calc_gini(self, y_vector:ndarray):
        """
        y_vector should be a column vector
        For a given column vector returns 
        1. Total number of samples
        2. Number of samples of each class
        3. Gini for that vector of targets
        """
        total_samples = y_vector.shape[0]
        each_class_sample = np.unique(y_vector, return_counts = True)[1]
        gini_coef = 1 - ((each_class_sample/each_class_sample.sum())**2).sum()
        return total_samples,each_class_sample,gini_coef #may be here we want to do some tests

    def predict(self, x_instance):
        pass