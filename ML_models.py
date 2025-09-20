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
    
    def __str__(self):
        return_string = ''
        double_space = '\n\n'
        #writing parameters
        t ='t: ' + str(self.data['t']) + '\n'
        class_ = 'class: ' + str(self.data['class']) + '\n'
        total_samples = 'total samples: ' + str(self.data['total_samples']) + '\n'
        class_samples = 'class samples: ' + str(self.data['each_class_samples']) + '\n'
        gini = 'Gini: ' + str(self.data['data_gini_coef']) + '\n'
        return_string += t
        return_string += class_
        return_string += total_samples
        return_string += class_samples
        return_string += gini
        return return_string






#models
class DecisionTreeClassifier:
    """
    implements a decision tree classifier
    """
    def __init__(self, max_depth = float('inf')):
        #Hyper parameters
        self.max_depth = max_depth
        #Fitting parameters
        self.best_class = None
        self.best_t = None
        self.data = dict()
        self.best_loss = float('inf')
        #The BinaryTree
        self.BinaryTree = TreeNode()

    def fit(self, X_data:ndarray, y_data:ndarray, node = None):
        """
        Fits the Decision Tree
        X has shape of N x M where M is number of features; N is number of instances 
        """
        #defining self parameters during training
        self.unique_classes = np.unique(y_data)
        #parameters for further training
        flag_added = False
        number_of_classes = X_data.shape[1]
        number_of_instances = X_data.shape[0]
        #Training algorithm
        if number_of_instances == 1:
            return self
        else:
            #main learning loop
            for feature_col in range(number_of_classes):
                for instance_row in range(number_of_instances):
                    #getting the t devider and masking for <= and > 
                    t_col = X_data[instance_row,feature_col]
                    t_mask = X_data[X_data[:,feature_col]] <= t_col
                    not_t_mask = X_data[X_data[:,feature_col]] > t_col
                    y_left = y_data[t_mask,:]
                    y_right = y_data[not_t_mask,:]
                    #getting the gini losses for both sides
                    total_samples_left, _, gini_coef_left = self.calc_gini(y_left)
                    total_samples_right, _, gini_coef_right = self.calc_gini(y_right)
                    total_instances = total_samples_right + total_samples_left
                    J_loss = total_samples_left/total_instances*gini_coef_left + total_samples_right/total_instances*gini_coef_right
                    if J_loss < self.best_loss:
                        #checking the necessary parameters
                        self.best_loss = J_loss
                        self.best_class = feature_col
                        self.best_t = t_col
                        flag_added = True
            #adding the necessary information to the tree
            data_total_samples, data_each_class_sample, data_gini_coef = self.calc_gini(y_data)
            self.data['t'] = self.best_t if flag_added else None
            self.data['class'] = self.best_class if flag_added else None
            self.data['total_samples'] = data_total_samples
            self.data['each_class_samples'] = data_each_class_sample
            self.data['data_gini_coef'] = data_gini_coef
            #saving data to main node
            main_node = self.BinaryTree if node == None else node
            main_node.data = self.data
            if flag_added == True:
                #if we found a better impurity we add the left and right branches
                node_left = TreeNode()
                node_right = TreeNode()
                main_node.left = node_left
                main_node.right = node_right
                self.fit(X_data[X_data[:,self.best_class] <= self.best_t,:], y_data[X_data[:,self.best_class] <= self.best_t,:], node = node_left)
                self.fit(X_data[X_data[:,self.best_class] > self.best_t,:], y_data[X_data[:,self.best_class] > self.best_t,:], node = node_right)

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
        return total_samples, each_class_sample, gini_coef #may be here we want to do some tests
    


    def predict(self, x_instance):
        pass