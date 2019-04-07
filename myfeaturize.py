'''
**************** PLEASE READ ***************

Script that reads in spam and ham messages and converts each training example
into a feature vector

Code intended for UC Berkeley course CS 189/289A: Machine Learning

Requirements:
-scipy ('pip install scipy')

To add your own features, create a function that takes in the raw text and
word frequency dictionary and outputs a int or float. Then add your feature
in the function 'def generate_feature_vector'

The output of your file will be a .mat file. The data will be accessible using
the following keys:
    -'training_data'
    -'training_labels'
    -'test_data'

Please direct any bugs to kevintee@berkeley.edu
'''

from collections import defaultdict
import glob
import re
import scipy.io
from sklearn.feature_extraction.text import TfidfVectorizer


NUM_TRAINING_EXAMPLES = 5172
NUM_TEST_EXAMPLES = 5857

BASE_DIR = './'
SPAM_DIR = 'spam/'
HAM_DIR = 'ham/'
TEST_DIR = 'test/'

# This method generates a design matrix with a list of filenames
# Each file is a single training example
def generate_design_matrix(filenames, vectorizer=None):
    if vectorizer is None:
        vectorizer = TfidfVectorizer(input='filename', analyzer='word', 
                                     decode_error='ignore', lowercase=True, max_df=0.6, max_features=500)
        return vectorizer.fit_transform(filenames), vectorizer 
    else:
        return vectorizer.transform(filenames)

# ************** Script starts here **************

spam_filenames = glob.glob(BASE_DIR + SPAM_DIR + '*.txt')
ham_filenames = glob.glob(BASE_DIR + HAM_DIR + '*.txt')
filenames = spam_filenames + ham_filenames
design_matrix, vectorizer = generate_design_matrix(filenames)
# Important: the test_filenames must be in numerical order as that is the
# order we will be evaluating your classifier
test_filenames = [BASE_DIR + TEST_DIR + str(x) + '.txt' for x in range(NUM_TEST_EXAMPLES)]
test_design_matrix = generate_design_matrix(test_filenames, vectorizer)

X = design_matrix.toarray()
Y = [1]*len(spam_filenames) + [0]*len(ham_filenames)

file_dict = {}
file_dict['training_data'] = X
file_dict['training_labels'] = Y
file_dict['test_data'] = test_design_matrix.toarray()
scipy.io.savemat('spam_data_BoW.mat', file_dict)

print(vectorizer.get_feature_names())
# print(vectorizer.stop_words_)


