# import packages
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree

#f = h5py.File('C:\Users\Packard Bell\Documents\MATLAB\All_Data.mat', 'r')
#data = f.get('data/xdata')
#data = np.array(data)

# read datasets
test = scipy.io.loadmat('All_Data.mat')

# Cross-Validation?
# Split datasets into train(first 80%), (validation) and test(last 20%), 80%65535 = 52428
# Train input(samples and features) and output(target) from datasets
x_train = [[test[:52428, 2]], [test[:52428, 3]], [test[:52428, 4]]]
y_train = [test[:52428, 26]]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)

# Test and predict class for remaining datasets (last 20%)
x_test = [[test[52429:, 2]], [test[52429:, 3]], [test[52429:, 4]]]
y_predict = clf.predict(x_test)

# Compare predicted value with actual target value (class)
# print predictions
for i in range(len(x_test)):
 print("x_test = %s, Prediction = %s" % (x_test[i], y_predict[i]))

# How good is the accuracy of the model?


# Adjust model with hyperparameters(regularization), avoid misclassifications and overfitting


# Test model on new data and evaluate the model


# print(test)
#tree.plot_tree(test)
