from collections import Counter

import numpy as np
from numpy import genfromtxt
import scipy.io
from scipy import stats

import random


class Node:

    def __init__(self, depth, left=None, right=None, label=None, is_leaf=False, 
                 split_rule=None, X=None, y=None):
        self.depth = depth
        self.left = left
        self.right = right
        self.label = label
        self.is_leaf = is_leaf
        self.split_rule = split_rule
        self.X = X
        self.y = y


class DecisionTree:

    def __init__(self):
        self.root = Node(0)

    @staticmethod
    def entropy(y):
        """
        Calculates the entropy given all the labels
        """
        n = len(y)
        if n <= 1:
            return 0
        _, counts = np.unique(y, return_counts=True)
        priors = counts / n
        entropy = 0
        for p in priors:
            entropy -= p * np.log2(p)
        return entropy
        

    @staticmethod
    def information_gain(X, y, thresh):
        """
        Calculates information gain given a feature vector
        (i.e, a column of the design matrix) and a split threshold
        """
        H = DecisionTree.entropy(y)
        
        Sl = [y[i] for i in range(len(y)) if X[i] < thresh]
        Sr = [y[i] for i in range(len(y)) if X[i] >= thresh]
        H_after = ((len(Sl) * DecisionTree.entropy(Sl) + len(Sr) * DecisionTree.entropy(Sr)) 
                    / len(y))        
        return H - H_after

    @staticmethod
    def gini_impurity(y):
        """
        Calculates the gini impurity given all the labels
        """
        n = len(y)
        if n <= 1:
            return 0
        _, counts = np.unique(y, return_counts=True)
        priors = counts / n
        summation = 0
        for p in priors:
            summation += p * p
        return 1 - summation

    @staticmethod
    def gini_purification(X, y, thresh):
        """
        Calculates reduction in impurity gain given a vector of features
        and a split threshold
        """
        I = DecisionTree.entropy(y)
        
        Sl = [y[i] for i in range(len(y)) if X[i] < thresh]
        Sr = [y[i] for i in range(len(y)) if X[i] >= thresh]
        I_after = ((len(Sl) * DecisionTree.gini_impurity(Sl) + len(Sr) * DecisionTree.gini_impurity(Sr)) 
                    / len(y))
        return I - I_after

    def split(self, X, y, idx, thresh):
        """
        Returns a split of the dataset given an index of the feature and
        a threshold for it
        """
        l_mask = X[:,idx] < thresh
        r_mask = ~l_mask
        return X[l_mask], y[l_mask], X[r_mask], y[r_mask]
    
    def segmenter(self, X, y):
        """
        Computes entropy gain for all single-dimension splits,
        return the feature and the threshold for the split that
        has maximum gain
        """
        # Initialize split rule arbitrarily
        split_idx = 0 
        split_thresh = X.T[0][0]
        max_gain = 0
        # Iterate through each feature
        for i, x in enumerate(X.T):
            # Iterate through each unique feature value
            thresh_vals = np.unique(x)
            for thresh in thresh_vals:
                gain = DecisionTree.information_gain(x, y, thresh)
                if gain > max_gain:
                    max_gain = gain
                    split_idx = i
                    split_thresh = thresh
        return split_idx, split_thresh
    
    def stop(self, node):
        if len(node.y) <= self.n_min or node.depth > self.d_max:
            return True
        return False
    
    def compute_label(self, node):
        return Counter(node.y).most_common(1)[0][0]
        
    
    def build_tree(self, node):
        if len(node.X) == 1 or self.stop(node):
            node.is_leaf = True
            node.label = self.compute_label(node)
            return
        node.split_rule = self.segmenter(node.X, node.y)
        Xleft, yleft, Xright, yright = self.split(node.X, node.y, node.split_rule[0], node.split_rule[1])
        # If best split doesn't yield any separation, end recursion
        if len(yleft) == 0 or len(yright) == 0:
            node.is_leaf = True
            node.label = self.compute_label(node)
        else:
            node.right = Node(node.depth + 1, X=Xright, y=yright)
            node.left = Node(node.depth + 1, X=Xleft, y=yleft)
            self.build_tree(node.left)
            self.build_tree(node.right)
            
        
    def fit(self, X, y, n_min=5, d_max=3):
        """
        fits the model to a training set. ...stopping criteria
        """
        # clear tree first, allows re-fitting
        self.root.left, self.root.right = None, None
        self.root.X = X
        self.root.y = y
        self.n_min = n_min
        self.d_max = d_max
        self.build_tree(self.root)
        
#         return 0

    def predict(self, X):
        """
        TODO: predict the labels for input data 
        """
        y_pred = []
        for x in X:
            node = self.root
            while not node.is_leaf:
                if x[node.split_rule[0]] < node.split_rule[1]:
                    node = node.left
                else:
                    node = node.right
            y_pred.append(node.label)
        return y_pred
    
    def tree_repr(self, node, level):
        if node is None:
            return None
        elif node.is_leaf:
            return "\t"*level+"label: "+str(node.label)+"\n"
        else:
            left = self.tree_repr(node.left, level + 1)
            right = self.tree_repr(node.right, level + 1)
            return "\t"*level+str(node.split_rule)+"\n" + left + right
#             return "\t"*level+repr(node.label)+"\n" + left + right

    def __repr__(self):
        """
        TODO: one way to visualize the decision tree is to write out a __repr__ method
        that returns the string representation of a tree. Think about how to visualize 
        a tree structure. You might have seen this before in CS61A.
        """
#         ret = "\t"+repr(self.root)+"\n"
        #TODO: write __repr__ for Node class
        assert(self.root.split_rule != None)
        ret = "\t"+str(self.root.split_rule)+"\n"
        ret += self.tree_repr(self.root.left, 2) + "left"
        ret += self.tree_repr(self.root.right, 2) + "right"
        return ret
    

        


class RandomForest():
    
    def __init__(self):
        """
        TODO: initialization of a random forest
        """

    def fit(self, X, y):
        """
        TODO: fit the model to a training set.
        """
        return 0
    
    def predict(self, X):
        """
        TODO: predict the labels for input data 
        """
        return 0

