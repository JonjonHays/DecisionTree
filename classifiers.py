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
        
    def to_string(self, feature_names):
        if self.is_leaf:
            return "label = " + str(self.label)
        return feature_names[self.split_rule[0]] + ": " + str(self.split_rule)


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
    
    @staticmethod
    def accuracy(y_pred, y_test):
        return sum(y_pred == y_test) / len(y_pred)

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
        if self.m is not None:
            feature_indcs = np.random.choice(range(X.shape[1]), self.m, replace=False) 
        else:
            feature_indcs = range(X.shape[1])
        # Iterate through each feature
        for i in feature_indcs:
            x = X[:,i]
            # Iterate through each unique feature value
            thresh_vals = np.unique(x)
            if len(thresh_vals) > self.max_thresh:
                thresh_vals = np.linspace(min(thresh_vals), max(thresh_vals), self.max_thresh)
            for thresh in thresh_vals:
                gain = DecisionTree.information_gain(x, y, thresh)
                if gain > max_gain:
                    max_gain = gain
                    split_idx = i
                    split_thresh = thresh
        return split_idx, split_thresh
    
    def stop(self, node):
        if len(node.y) <= self.purity_max or node.depth > self.d_max:
            return True
        return False
    
    def compute_label(self, node):
        return Counter(node.y).most_common(1)[0][0]
        
    def build_tree(self, node, max_thresh=10):
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
            
        
    def fit(self, X, y, purity_max=5, d_max=3, max_thresh=10, m=None):
        """
        fits the model to a training set. ...stopping criteria
        """
        # clear tree first, allows re-fitting
        self.root.left, self.root.right = None, None
        self.root.X = X
        self.root.y = y
        self.purity_max = purity_max
        self.d_max = d_max
        self.max_thresh = max_thresh
        self.m = m
        self.build_tree(self.root)


    def predict(self, X):
        """
        predict the labels for input data 
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
    
    
    def predict_trace(self, x, feature_names=None, label_names=None):
        """
        Predict the label of a sample point, and trace the decisions
        made to reach the final prediction
        """
        node = self.root
        while not node.is_leaf:
            if feature_names is None:
                name = str(node.split_rule[0])
            else:
                name = feature_names[node.split_rule[0]]
            if x[node.split_rule[0]] < node.split_rule[1]:
                print("\'" + name + "\'" + " < " + str(node.split_rule[1]))
                node = node.left
            else:
                print("\'" + name + "\'" + " >= " + str(node.split_rule[1]))
                node = node.right
        if label_names is not None:
            lname = label_names[node.label]
        else:
            lname = str(node.label)
        print("Therefore this email is " + lname)
        y_pred = node.label
    
    def print_tree(self, root, feature_names, indent=0):
        """
        Print a representation of the decision tree
        """
        if root is None:
            return
        print('    ' * indent + root.to_string(feature_names))
        for node in [root.left, root.right]: 
            self.print_tree(node, feature_names, indent + 1)
        


class RandomForest():
    
    def __init__(self, random_seed=13):
        np.random.seed(random_seed)
        self.trees = []

    def fit(self, X, y, n_trees=10, n_samples=None, m=None, purity_max=5, d_max=3, max_thresh=10):
        """
        Fit the model to a training set.
        """
        if m is None:
            m = int(np.sqrt(X.shape[1]))
        if m > X.shape[1]:
            m = X.shape[1]
        if n_samples is None:
            n_samples = X.shape[0]
        for _ in range(n_trees):
            d = DecisionTree()
            # Sample n points with replacement to yield subsample n' 
            rand_indcs = np.random.choice(range(X.shape[0]), n_samples, replace=True)
            Xsub = X[rand_indcs]
            ysub = y[rand_indcs]
            d.fit(Xsub, ysub, purity_max=purity_max, d_max=d_max, max_thresh=max_thresh, m=m)
            self.trees.append(d)
    
    def predict(self, X):
        """
        TODO: predict the labels for input data 
        """
        y_preds = []
        for tree in self.trees:
            y_preds.append(tree.predict(X))
        y_preds = np.array(y_preds)
        y_pred = []
        for i in range(y_preds.shape[1]):
            y_pred.append(Counter(y_preds[:,i]).most_common(1)[0][0]) 
        return y_pred
    


