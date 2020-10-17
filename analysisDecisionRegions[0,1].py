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
print(iris.head())
features = ['sepal length', 'sepal width', 'petal length', 'petal width', 'category']
iris.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'category']
irisColumns = [iris['sepal length'], iris['sepal width'], iris['petal length'], iris['petal width']]
categories = ['Setosa', 'Versicolor', 'Virginica']
iris['catNB'] = pd.factorize(iris['category'].values)[0]
irisLabels = iris['catNB'] 
#irisLabels = iris['category'] 
iris = iris.drop(labels='category', axis=1)

model = GaussianNB()

currentDF = pd.DataFrame()
i = 0 # First Column
j = 1 # Second Column
currentDF.insert(0, features[i], irisColumns[i].to_list(), True)
currentDF.insert(1, features[j], irisColumns[j].to_list(), True)
model.fit(currentDF, irisLabels)

# Loading some example data
iris = datasets.load_iris()
X = iris.data[:, [0,1]]
y = iris.target

# Plotting decision regions
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1 #Compute boundaries of painting space
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1)) #tesselation 0.1 - resolution

f, ax = plt.subplots(figsize=(10, 8))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

sns.set_theme(style="white")

iris2 = pd.read_csv('Iris.csv', header = None)
iris2.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'category']
print(iris2.head())
colors=['green', 'blue', 'red']
markers = ['s', 'p', 'P']
cmap = ListedColormap(colors)
ax = sns.scatterplot(data=iris2, x='sepal length', y='sepal width', hue = "category", palette=colors, style="category", markers=markers)
ax.contourf(xx, yy, Z, alpha=0.2, cmap=cmap) # Paint between boarders

#scatter = ax.scatter(x, y, c=c, s=s)
#ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k') # Crosses and Pluses, etc.

#legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")

#ax.add_artist(legend1)

ax.set_title('Naive Bayes Classifier trained on [0,1] Decision Regions')

plt.show()