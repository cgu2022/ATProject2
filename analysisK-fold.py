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
from sklearn.model_selection import KFold

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
for k in range(3, 10):
    print("\nK:", k)
    kFolds = KFold(n_splits = k, shuffle = True, random_state = 0)
    i = 0
    for trainIdx, testIdx in kFolds.split(x):
        xtrain, ytrain = x[trainIdx], y[trainIdx]
        xtest, ytest = x[testIdx], y[testIdx]
        model = GaussianNB()
        model.fit(xtrain, ytrain)
        currentScore = model.score(xtest, ytest)
        print("Accuracy for test " + str(i) +":", currentScore)
        accuracy += currentScore
        predicted = model.predict(xtest)
        i += 1
        # confusion += confusion_matrix(ytest, predicted)
    accuracy /= k
    print("Mean Score:", accuracy)
    accuracy = 0
