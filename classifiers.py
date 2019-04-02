from collections import Counter

import numpy as np
from numpy import genfromtxt
import scipy.io
from scipy import stats

import random


class Node:

    def __init__(self, left, right, label, is_leaf=False, split_rule=None):
        self.left = left
        self.right = right
        self.label = label
        self.is_leaf = is_leaf
        self.split_rule = split_rule



class DecisionTree:

    def __init__(self):
        self.root = Node()

    @staticmethod
    def entropy(y):
        """
        TODO: implement a method that calculates the entropy given all the labels
        """
        print("entered!")
        classes = {}
        for c in y:
            if classes.get(c):
                classes[c] += 1
            else:
                classes[c] = 1
        entropy = 0
        n = len(classes.keys())
        for c in classes.keys():
            p = classes[c] / n
            surprise = -np.log2(p)
            entropy += p * surprise
        return entropy

    @staticmethod
    def information_gain(X, y, thresh):
        """
        TODO: implement a method that calculates information gain given a vector of features
        and a split threshold
        """
        return 0

    @staticmethod
    def gini_impurity(y):
        """
        TODO: implement a method that calculates the gini impurity given all the labels
        """
        return 0

    @staticmethod
    def gini_purification(X, y, thresh):
        """
        TODO: implement a method that calculates reduction in impurity gain given a vector of features
        and a split threshold
        """
        return 0

    def split(self, X, y, idx, thresh):
        """
        TODO: implement a method that return a split of the dataset given an index of the feature and
        a threshold for it
        """
        return 0
    
    def segmenter(self, X, y):
        """
        TODO: compute entropy gain for all single-dimension splits,
        return the feature and the threshold for the split that
        has maximum gain
        """
        return 0
    
    
    def fit(self, X, y):
        """
        TODO: fit the model to a training set. Think about what would be 
        your stopping criteria
        """
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

