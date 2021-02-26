# import packages
import scipy.io
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn import tree

# read datasets
test = scipy.io.loadmat('All_Data.mat')

# identify input and output from datasets
X = [[test[:, 2]], [test[:, 3]], [test[:, 4]]]
Y = [test[:, 26]]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

# now predict class of datasets
y_predict = clf.predict(X)


# print(test)
#xdata=test['x']
#ydata=test['y']
#zdata=test['z']

#fig = plt.figure()
#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(xdata, ydata, zdata, rstride=1, cstride=1, cmap=cm.jet)
#fig.colorbar(surf)

#plt.show()

#cp = plt.pcolor(xdata, ydata, zdata, cmap='jet',vmin=0,vmax=5000)
#plt.colorbar(cp)
#plt.title('firstsets')
#plt.xlabel('x-label')
#plt.ylabel('y-label')
