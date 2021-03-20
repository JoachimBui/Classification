# import packages
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import h5py
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix

#f = h5py.File('C:\Users\Packard Bell\Documents\MATLAB\All_Data.mat', 'r')
#data = f.get('data/xdata')
#data = np.array(data)

# Read datasets
test = scipy.io.loadmat('All_Data.mat')

# Cross-Validation? "train_test_split()" function?
# Split datasets into train(first 80%), (validation) and test(last 20%)
# Train input(samples and features) and output(target) from datasets
x_train = [[test[:52428, 2]], [test[:52428, 3]], [test[:52428, 4]]]   # 80%65535 = 52428
y_train = [test[:52428, 26]]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)

# Test and predict class for remaining datasets (last 20%)
x_test = [[test[52429:, 2]], [test[52429:, 3]], [test[52429:, 4]]]
y_predict = clf.predict(x_test)

# Compare predicted value with actual target value (class/labels)
# How good is the accuracy of the model?
y_test = [test[52429:, 26]]
print(confusion_matrix(y_test, y_predict))   # Matrix showing all the correct and incorrect predictions
print(classification_report(y_test, y_predict))    # Precision, recall, F1-score and support

# Print predictions
#for i in range(len(x_test)):
 #print("x_test = %s, Prediction = %s" % (x_test[i], y_predict[i]))

 
# Adjust model with hyperparameters(regularization), avoid misclassifications and overfitting


# Test model on new data and evaluate the model


# Print(test)
#tree.plot_tree(test)
