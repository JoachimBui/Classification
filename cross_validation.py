# Import packages
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

# Load datasets
test = scipy.io.loadmat('All_Data.mat')

# Initialize features, target/labels and divide model into train and test
X = test['All_Data'][:, 2:5]
Y = test['All_Data'][:, 26]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Hyperparameter tuning
# Try different range of values
# Number of trees
n_estimators = [10, 20, 50, 100, 200]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Max number of levels in tree
max_depth = [10, 20, 30, 40, 50]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 20]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10, 15]

# Create dictionary to store the hyperparameters
param_grid = { 'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

rf = RandomForestClassifier()

# RandomSearchCV to sample random range of values, then fit and evaluate model
#rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=100, cv=5, verbose=2, n_jobs=-1, random_state=42)
#rf_random.fit(X_train, y_train)
#print(rf_random.best_params_)
#print(rf_random.best_score_)

# Use the gridSearchCV function to cross validate and find the best values for each hyperparameter
rf_grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1)
rf_grid.fit(X_train, y_train)
print(rf_grid.best_params_)
print(rf_grid.best_score_)

