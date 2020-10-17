from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import *
from sklearn.naive_bayes import *

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

iris = pd.read_csv('Iris.csv', header = None)
#print(iris.head())
features = ['sepal length', 'sepal width', 'petal length', 'petal width', 'category']
iris.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'category']
irisColumns = [iris['sepal length'], iris['sepal width'], iris['petal length'], iris['petal width']]
categories = ['Setosa', 'Versicolor', 'Virginica']
iris['catNB'] = pd.factorize(iris['category'].values)[0]
irisLabels = iris['catNB'] 
#iris = iris.drop(labels='category', axis=1)

model = GaussianNB()

currentDF = pd.DataFrame()
i = 0 # First Column
j = 2 # Second Column
k = 3
currentDF.insert(0, features[i], irisColumns[i].to_list(), True)
currentDF.insert(1, features[j], irisColumns[j].to_list(), True)
currentDF.insert(2, features[k], irisColumns[k].to_list(), True)
print(currentDF)
model.fit(currentDF, irisLabels)

# Calculating PCA
'''x = iris.iloc[:,:4].values      # x is now an Numpy array
covMat = np.cov(x.T)            # covMat is (4,4)
eigenValues, eigenVectors = np.linalg.eig(covMat)
idx = eigenValues.argsort()[::-1]
eigenValues = eigenValues[idx]
eigenVectors = eigenVectors[:,idx]
projMat = eigenVectors[:,:2]
xP = x.dot(projMat)
'''

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)

x = currentDF.values
pca.fit(x)
print('Components using Scikit-learn =')
print(pca.singular_values_)
print(pca.components_)
xP = pca.transform(x)

projIris = pd.DataFrame(data = xP, columns = ['eig1', 'eig2'])

# Loading some example data
X = projIris
y = irisLabels

# Training classifiers

X = X.to_numpy()
print("X:", X)
# Plotting decision regions
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1 #Compute boundaries of painting space
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01)) #tesselation 0.1 - resolution

f, ax = plt.subplots(figsize=(10, 8))

print(np.c_[xx.ravel(), yy.ravel()].shape)

Z = model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

sns.set_theme(style="white")

iris2 = pd.read_csv('Iris.csv', header = None)
iris2.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'category']
#print(iris2.head())
colors=['green', 'blue', 'red']
markers = ['s', 'p', 'P']
cmap = ListedColormap(colors)

projIris['category'] = iris2['category']
print(projIris.head())
ax = sns.scatterplot(data=projIris, x='eig1', y='eig2', hue = "category", palette=colors, style="category", markers=markers)
ax.contourf(xx, yy, Z, alpha=0.2, cmap=cmap) # Paint between boarders

#scatter = ax.scatter(x, y, c=c, s=s)
#ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k') # Crosses and Pluses, etc.

#legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")

#ax.add_artist(legend1)
ax.set_title('NBC trained on [0,2,3] Decision Regions plotted on PCA Space')

plt.show()