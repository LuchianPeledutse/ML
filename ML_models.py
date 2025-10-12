#imports 
import typing
from collections import abc

import numpy as np
from tqdm import tqdm


#regularizing parameters we want to implement
#max_depth
#min_samples
#min_leafs
#min % branch split


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
        class_ = 'class: ' + str(self.data['class']) + '\n'
        total_samples = 'total samples: ' + str(self.data['total_samples']) + '\n'
        class_samples = 'class samples: ' + str(self.data['each_class_samples']) + '\n'
        gini = 'Gini: ' + str(self.data['data_gini_coef']) + '\n'
        loss = 'J_loss: ' + str(self.data['loss']) + '\n'
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
    def __init__(self, criterion: abc.Callable[[object], float], max_depth = None):
        #overall parameters
        self.depth_count = False
        #Hyper parameters
        self.max_depth = max_depth
        self.criterion = criterion
        #The BinaryTree
        self.BinaryTree = TreeNode()

    def fit(self, X_data:np.ndarray, y_data:np.ndarray, node = None, best_class = None, best_t = None):
        """
        Fits the Decision Tree
        X has shape of N x M where M is number of features; N is number of instances 
        """
        #saving number of classes
        if self.rec_flag:
            self.classes = np.unique(y_data).tolist()
        #parameters for training
        data = dict()
        flag_added = False
        number_of_classes = X_data.shape[1]
        number_of_instances = X_data.shape[0]
        #Training algorithm
        if number_of_instances == 1:
            _, data_each_class_sample, _= self.calc_gini(y_data)
            data = {
                't': 'None',
                'class': 'None',
                'total_samples': 1,
                'each_class_samples': data_each_class_sample,
                'data_gini_coef': 0.0,
                'loss': 0.0
                }
            main_node = self.BinaryTree if node == None else node
            main_node.data = data
        else:
            #main learning loop
            #fixating the branch splitting function
            best_loss = float('inf')
            for feature_col in range(number_of_classes):
                x_feature_col_data = np.sort(X_data[:,feature_col]).reshape(-1,1)
                for instance_row in range(number_of_instances-1):
                    #getting the t devider and masking for <= and > boolean predicates 
                    t_crit = (x_feature_col_data[instance_row,:] + x_feature_col_data[instance_row+1,:])/2 
                    t_mask = x_feature_col_data.reshape(-1) <= t_crit.item()
                    not_t_mask = x_feature_col_data.reshape(-1) > t_crit.item()
                    y_left = y_data[t_mask,:]
                    y_right = y_data[not_t_mask,:]
                    #getting the gini losses for both sides
                    total_samples_left, _, gini_coef_left = self.calc_gini(y_left)
                    total_samples_right, _, gini_coef_right = self.calc_gini(y_right)
                    total_instances = total_samples_right + total_samples_left
                    J_loss = total_samples_left/total_instances*gini_coef_left + total_samples_right/total_instances*gini_coef_right
                    if J_loss < best_loss and 0 not in [total_samples_left, total_samples_right]:
                        #checking the necessary parameters
                        best_loss = J_loss
                        best_class = feature_col
                        best_t = t_crit
                        flag_added = True
            #adding the necessary information to the tree
            data_total_samples, data_each_class_sample, data_gini_coef = self.calc_gini(y_data)
            data['t'] = best_t if flag_added else 'None'
            data['class'] = best_class if flag_added else 'None'
            data['total_samples'] = data_total_samples
            data['each_class_samples'] = data_each_class_sample
            data['data_gini_coef'] = data_gini_coef
            data['loss'] = best_loss
            #saving data to main node
            main_node = self.BinaryTree if node == None else node
            main_node.data = data
            # print(main_node)
            if flag_added == True and data_gini_coef != 0.0:
                #add recursion count
                self.rec_flag = False
                #if we found a better impurity we add the left and right branches
                node_left, node_right = TreeNode(), TreeNode()
                main_node.left = node_left
                main_node.right = node_right
                self.fit(X_data[X_data[:,best_class]<=best_t,:], y_data[X_data[:,best_class]<=best_t,:], node = node_left)
                self.fit(X_data[X_data[:,best_class]>best_t,:], y_data[X_data[:,best_class]>best_t,:], node = node_right)

    def calc_gini(self, y_vector:np.ndarray):
        """
        parameters
        -----------
        y_vector: np.ndarray of shape (N,1) (i.e. vector column)

        returns
        --------
        1. Total number of samples
        2. Number of samples of each class
        3. Gini for that vector of targets
        """
        cls_dict = dict(zip(self.classes,[0 for _ in range(len(self.classes))]))
        total_samples = y_vector.shape[0]
        each_class_sample = np.unique(y_vector, return_counts = True)
        #updating count values in main dict
        for cls, count in zip(each_class_sample[0],each_class_sample[1]):
            cls_dict[cls] += count.item()
        #getting the stats for all classes
        samples_for_all = np.array(list(cls_dict.values()))
        gini_coef = 1 - ((each_class_sample[1]/samples_for_all.sum())**2).sum()
        return total_samples, samples_for_all, gini_coef #may be here we want to do some tests
    


    def predict(self, X):
        """
        Input: X is a matrix of shape N_samples x M_features
        Output: Y of shape N_samples x 1 of predicted features
        """
        #preparing parameters
        N_instances = X.shape[0]
        y_prediction = np.zeros((N_instances,1))
        for row in range(N_instances):
            initial_node = self.BinaryTree
            while initial_node.right != None:
                t = initial_node.data['t']
                cls_col = initial_node.data['class']
                if X[row,cls_col] <= t:
                    initial_node = initial_node.left
                elif X[row,cls_col] > t:
                    initial_node = initial_node.right
            cls_indx_pred = initial_node.data['each_class_samples'].argmax(axis = 0).item()
            y_pred = self.classes[cls_indx_pred]
            y_prediction[row,0] = y_pred
        return y_prediction


        