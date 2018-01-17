from sklearn import tree
# Tree Specific submodule of Scikit learn which will let us build a machine learning of a decision tree.
#It will every point in the tree the more datapoints it receives. An unlabelled data point can be fed into the tree, it will ask a series of question until it labels it. Label is our classification; the more data we #trade it on the more accurate the classification is 

from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
    [190, 90, 47], [175, 64, 39],[177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

# [height, weight, shoe_size]
# X lists of lists
# List is a datatype in python that can store a sq. of values

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']
# Y--> list of labels (Each label is a gender associated with a body metrics in X) , Here we have written it as strings (a datatype used to represent
#text instead of numbers)


# Classifiers
# using the default values for all the hyperparameters
#clf_tree,clf_svm,clf_perceptron (3 classifiers)--> 3 Variables to store the respective decision tree models

clf_tree = tree.DecisionTreeClassifier()
# Calling 'decision tree' method on the 'Tree' object

clf_svm = SVC()
clf_perceptron = Perceptron()
clf_KNN = KNeighborsClassifier()

# Training the models
clf_tree = clf_tree.fit(X, Y)
# Calling fit method on the classifier method
# 'fit' method trains the decision tree on our dataset
# Takes 2 arguments X,Y and the result will be stored in the updated 'clf' variable

clf_svm = clf_svm.fit(X, Y)
clf_perceptron = clf_perceptron.fit(X, Y)
clf_KNN = clf_KNN.fit(X, Y)

# We are testing the classifier with a new list of body metrics using 3 different classifiers
prediction_tree = clf_tree.predict([[190, 70, 43]])
prediction_svm = clf_svm.predict([[190, 70, 43]])
prediction_perceptron = clf_perceptron.predict([[190, 70, 43]])
prediction_KNN = clf_KNN.predict([[190, 70, 43]])

# Printing the predictions
print(prediction_tree)
print(prediction_svm)
print(prediction_perceptron)
print(prediction_KNN)

# Testing using the same data and predicting their accuracy
pred_tree = clf_tree.predict(X)
acc_tree = accuracy_score(Y, pred_tree) * 100
print('Accuracy for DecisionTree: {}'.format(acc_tree))

pred_svm = clf_svm.predict(X)
acc_svm = accuracy_score(Y, pred_svm) * 100
print('Accuracy for SVM: {}'.format(acc_svm))

pred_per = clf_perceptron.predict(X)
acc_per = accuracy_score(Y, pred_per) * 100
print('Accuracy for perceptron: {}'.format(acc_per))

pred_KNN = clf_KNN.predict(X)
acc_KNN = accuracy_score(Y, pred_KNN) * 100
print('Accuracy for KNN: {}'.format(acc_KNN))

# The best classifier from svm, per, KNN
index = np.argmax([acc_svm, acc_per, acc_KNN])
classifiers = {0: 'SVM', 1: 'Perceptron', 2: 'KNN'}
print('Best gender classifier is {}'.format(classifiers[index]))
