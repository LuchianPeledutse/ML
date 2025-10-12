#imports 
import typing
from collections import abc

import numpy as np
from tqdm import tqdm


#data node structure for tree regressor
class TreeNode(object):
    def __init__(self, data:abc.Mapping[str,typing.Any]|None = None):
        self.data = data
        self.left = None
        self.right = None
    
    def __str__(self):
        return_string = ''
        double_space = '\n\n'
        #writing parameters
        t ='t: ' + str(self.data['t']) + '\n'
        class_ = 'split_feature: ' + str(self.data['split_feature']) + '\n'
        total_samples = 'total samples: ' + str(self.data['total_samples']) + '\n'
        class_samples = 'class samples: ' + str(self.data['each_class_samples']) + '\n'
        gini = 'information_criterion: ' + str(self.data['information_criterion']) + '\n'
        loss = 'split_criterion: ' + str(self.data['split_criterion']) + '\n'
        return_string += t
        return_string += class_
        return_string += total_samples
        return_string += class_samples
        return_string += loss
        return_string += gini
        return return_string
    

class GiniCriterion(object):
    """Implements Gini criterion"""
    def __init__(self):
        #dict for counting classes
        self.dict_cls_count = dict()

    def __call__(self, target:np.ndarray) -> float:
        """
        parameters
        ----------
        target: np.ndarray of shape Nx1 containing integers (i.e. integer column vector representing classes)

        returns
        -------
        Gini criterion for target dataset
        """
        #can we implement that logic with decorator?
        self.update_cls_dict(target)
        #getting dictionary values that represent counts for each class
        cls_counts = self.dict_cls_count.values()
        #O(N) for summing
        total_count_sum = sum(cls_counts)
        #O(N) for calculating gini
        gini_crit = 1
        for one_count in cls_counts:
            gini_crit -= (one_count/total_count_sum)**2
        return gini_crit
                


    def update_cls_dict(self, target:np.ndarray) -> None:
        """
        parameters
        ----------
        target: np.ndarray of shape Nx1 containing integers (i.e. integer column vector representing classes)

        returns
        -------
        None

        description
        ------------
        updates counts of classes in object dictionary
        """
        #clear dict from past calculations 
        self.dict_cls_count.clear()
        #counting classes (O(N) complexity for N samples in target)
        for one_cls in target:
            try:
                self.dict_cls_count[one_cls.item()] += 1
            except:
                self.dict_cls_count[one_cls.item()] = 1





#models
class DecisionTreeClassifier(object):
    """
    Implements a DecisionTreeClassifier
    """
    def __init__(self, criterion: abc.Callable[[np.ndarray], float] = GiniCriterion(), max_depth:int = None):
        #Training parameters (i.e. parameters needed for training algorithm)
        self.depth_count = False
        self.left_depth_count = False
        self.right_depth_count = False
        #Hyper parameters of model
        self.max_depth = max_depth
        self.min_samples = None
        self.min_leafs = None
        self.criterion = criterion
        #The BinaryTree
        self.BinaryTree = TreeNode()

    def fit(self, X_data:np.ndarray, y_data:np.ndarray, node:TreeNode = None, best_feature_col:int = None, best_t:float = None):
        """
        parameters
        -----------
        X_data: np.ndarray | shape NxM where N number of samples, M number of features
        y_data: np.ndarray | shape Nx1 is target array of classes
        node: TreeNode | current node if not None else root (self.BinaryTree) of tree
        best_feature_col: int | index of column (feature) that splits the data optimally
        best_t: float | the optimal value for boolean split on best_feature_col
        """
        #Training variables and statistics
        data = dict()
        flag_added = False
        best_split_criterion = float('inf')
        number_of_features = X_data.shape[1]
        number_of_instances = X_data.shape[0]
        #Training algorithm
        if number_of_instances == 1:
            information_criterion = self.criterion(y_data)
            data = {
                't': 'None',
                'feature_column': 'None',
                'total_samples': 1,
                'each_class_samples': list(self.criterion.dict_cls_count.items()),
                'information_criterion': 0.0,
                'split_criterion': 0.0
                }
            main_node = self.BinaryTree if node == None else node
            main_node.data = data
        else:
            #Main learning loop
            for feature_col in range(number_of_features):
                for instance_row in range(number_of_instances-1):
                    #getting splitting value t and its corresponding masks
                    t_value = (X_data[instance_row,feature_col].item() + X_data[instance_row+1,feature_col].item())/2 
                    t_mask = X_data[:,feature_col] <= t_value
                    not_t_mask = X_data[:,feature_col] > t_value
                    #getting right and left corresponding target data
                    y_left = y_data[t_mask,:]
                    y_right = y_data[not_t_mask,:]
                    #getting criterion of branch splitting for both sides (O(N) complexity)
                    criterion_left = self.criterion(y_left)
                    criterion_right = self.criterion(y_right)
                    #calculating split criterion (intermidiate lengths calculations have overall complexity O(N))
                    split_criterion = len(y_left)/number_of_instances*criterion_left + len(y_right)/number_of_instances*criterion_right
                    if split_criterion < best_split_criterion:
                        #updating training variables
                        best_split_criterion = split_criterion
                        best_feature_col = feature_col
                        best_t = t_value
                        flag_added = True
            #adding information to current node
            information_criterion = self.criterion(y_data)
            #mask for adding splitting information
            splitting_node = (flag_added and information_criterion != 0.0)
            data['t'] = best_t if splitting_node else 'None'
            data['split_feature'] = best_feature_col if splitting_node else 'None'
            data['total_samples'] = sum(self.criterion.dict_cls_count.values())
            data['each_class_samples'] = list(self.criterion.dict_cls_count.items())
            data['information_criterion'] = information_criterion
            data['split_criterion'] = best_split_criterion
            #saving data to main node
            main_node = self.BinaryTree if node == None else node
            main_node.data = data
            #if we have a split add recursion, given that data is not pure (i.e. more than one class)
            if splitting_node:
                node_left, node_right = TreeNode(), TreeNode()
                main_node.left = node_left
                main_node.right = node_right
                self.fit(X_data[X_data[:,best_feature_col]<=best_t,:], y_data[X_data[:,best_feature_col]<=best_t,:], node = node_left)
                self.fit(X_data[X_data[:,best_feature_col]>best_t,:], y_data[X_data[:,best_feature_col]>best_t,:], node = node_right)

    def predict(self, X:np.ndarray) -> np.ndarray:
        """
        parameters
        -----------
        X: np.ndarray | matrix of shape N_instances x M_features

        returns
        --------
        Y: np.ndarray | vector column of shape N_instances x 1 (i.e. prediction for each object in X)
        """
        #statistics
        N_instances = X.shape[0]
        target_prediction = np.zeros((N_instances,1))
        for row in range(N_instances):
            initial_node = self.BinaryTree
            #while the node is splitted
            while initial_node.right != None:
                t = initial_node.data['t']
                feature_col = initial_node.data['split_feature']
                #object can go either to left or right side depending on boolean value of predicate
                if X[row,feature_col] <= t:
                    initial_node = initial_node.left
                elif X[row,feature_col] > t:
                    initial_node = initial_node.right
            #get the most frequent class as prediction
            class_prediction = max(initial_node.data['each_class_samples'], key = lambda x: x[1])[0]
            target_prediction[row,0] = class_prediction
        return target_prediction


        