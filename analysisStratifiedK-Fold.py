from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import *
from sklearn.naive_bayes import *
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn import datasets

# Read in the Data
iris = pd.read_csv('Iris.csv', header = None)
iris.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'category']
categories = ['Setosa', 'Versicolor', 'Virginica']
irisLabels = iris['category'] 
iris = iris.drop(labels='category', axis=1)
print(irisLabels, end='\n\n\n')
print(iris) 

x = iris.iloc[:,:4].values
y = irisLabels

accuracy = 0.0
confusion = np.zeros((3,3))
k = 10
model = GaussianNB()
scores = cross_val_score(model, x, y, cv = k)
predicted = cross_val_predict(model, x, y, cv = k)
accuracy = scores.mean()
confusion = confusion_matrix(y, predicted)
confDF = pd.DataFrame(confusion)
confDF.columns = categories
confDF.index = categories

print("Accuracy:",accuracy)

# Heatmap & Confusion Matrix
fig, ax = plt.subplots() # figsize=(6,10)
plt.title('Confusion matrix of Stratified K-Fold (in total)')
ax = sns.heatmap(confDF, annot = True, square=True, linewidth=3)
fig.savefig('Heatmap of Stratified K-Fold.png')