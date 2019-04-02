from collections import Counter

import numpy as np
from numpy import genfromtxt
import scipy.io
from scipy import stats

import random


class Node:

    def __init__(self, left=None, right=None, label=None, is_leaf=False, 
                 split_rule=None, X=None, y=None):
        self.left = left
        self.right = right
        self.label = label
        self.is_leaf = is_leaf
        self.split_rule = split_rule
        self.X = X
        self.y = y



class DecisionTree:

    def __init__(self):
        self.root = Node(is_leaf=True)

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
                   
#                    (len(Sl) + len(Sr)))
        
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
                   
#         (len(Sl) + len(Sr)))
        
        return I - I_after

    def split(self, X, y, idx, thresh):
        """
        Returns a split of the dataset given an index of the feature and
        a threshold for it
        """
        l_indcs = [j for j in range(len(y)) if X[:,idx][j] < thresh]
        r_indcs = [j for j in range(len(y)) if X[:,idx][j] >= thresh]
        return X[l_indcs], y[l_indcs], X[r_indcs], y[r_indcs]
    
    def segmenter(self, X, y, entropy_gain=DecisionTree.information_gain):
        """
        Computes entropy gain for all single-dimension splits,
        return the feature and the threshold for the split that
        has maximum gain
        """
        # Initialize split rule arbitrarily
        split_idx = X.T[0]
        split_thresh = X.T[0][0]
        max_gain = 0
        # Iterate through each feature
        for i, x in enumerate(X.T):
            # Iterate through each unique feature value
            thresh_vals = np.unique(x)
            for thresh in thresh_vals:
                gain = entropy_gain(x, y, thresh)
                if gain > max_gain:
                    max_gain = gain
                    split_idx = i
                    split_thresh = thresh
        return split_idx, split_thresh
    
    
    def fit(self, X, y):
        """
        fits the model to a training set. ...stopping criteria
        """
        
        # clear tree first
        return 0

    def predict(self, X):
        """
        TODO: predict the labels for input data 
        """
        return 0

    def __repr__(self):
        """
        TODO: one way to visualize the decision tree is to write out a __repr__ method
        that returns the string representation of a tree. Think about how to visualize 
        a tree structure. You might have seen this before in CS61A.
        """
        return 0


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

