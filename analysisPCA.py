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
irisLabels = iris['category'] 

# PCA
x = iris.iloc[:,:4].values      # x is now an Numpy array
covMat = np.cov(x.T)            # covMat is (4,4)
eigenValues, eigenVectors = np.linalg.eig(covMat)
idx = eigenValues.argsort()[::-1]
eigenValues = eigenValues[idx]
eigenVectors = eigenVectors[:,idx]
print('Components using Numpy =')
print(eigenValues)
print(eigenVectors)
projMat = eigenVectors[:,:2]
xP = x.dot(projMat)

# Naive Bayes Classifier
model = GaussianNB()
model.fit(xP, irisLabels)
accuracy = model.score(xP, irisLabels)
predicted = model.predict(xP)
confusion = confusion_matrix(irisLabels, predicted)
confDF = pd.DataFrame(confusion)
confDF.columns = categories
confDF.index = categories

print("Accuracy:",accuracy)

# Heatmap & Confusion Matrix
fig, ax = plt.subplots() # figsize=(6,10)
plt.title('Confusion matrix with features with two largest eigenvectors ')
ax = sns.heatmap(confDF, annot = True, square=True, linewidth=3)
fig.savefig('Heatmap of NBC trained on PCA Features.png')

X = xP
y = iris['catNB']

# Training classifiers
model = GaussianNB()
model.fit(X, y)

print("X:", X)
# Plotting decision regions
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1 # Compute boundaries of painting space
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1)) # tesselation 0.1 - resolution

f, ax = plt.subplots(figsize=(10, 8))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

#print(iris2.head())
colors=['green', 'blue', 'red']
markers = ['s', 'p', 'P']
cmap = ListedColormap(colors)

xP = pd.DataFrame(data=xP, columns=["eig1", "eig2"]) # Convert Numpy array to Dataframe
iris2 = pd.read_csv('Iris.csv', header = None)
iris2.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'category']
xP['category'] = iris2['category'] 

print(xP.head())

ax = sns.scatterplot(data=xP, x='eig1', y='eig2', hue = "category", palette=colors, style="category", markers=markers)
ax.contourf(xx, yy, Z, alpha=0.2, cmap=cmap) # Paint between boarders

ax.set_title('NBC trained on PCA Decision Regions')

plt.show()