import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import *
from sklearn.naive_bayes import *

iris = pd.read_csv('Iris.csv', header = None)
features = ['sepal length', 'sepal width', 'petal length', 'petal width', 'category']
iris.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'category']
categories = ['Setosa', 'Versicolor', 'Virginica']
irisLabels = iris['category'] 
iris = iris.drop(labels='category', axis=1)

model = GaussianNB()

irisColumns = [iris['sepal length'], iris['sepal width'], iris['petal length'], iris['petal width']]
filePtr = open("Heatmaps/accuracies.txt", 'w+')

for i in range(0, 4):
    for j in range(i+1, 4):
        currentDF = pd.DataFrame()
        currentDF.insert(0, features[i], irisColumns[i].to_list(), True)
        currentDF.insert(1, features[j], irisColumns[j].to_list(), True)
        #print(currentDF)
        model.fit(currentDF, irisLabels)
        accuracy = model.score(currentDF, irisLabels)
        predicted = model.predict(currentDF)
        confusion = confusion_matrix(irisLabels, predicted)
        confDF = pd.DataFrame(confusion)
        confDF.columns = categories
        confDF.index = categories

        filePtr.write("Features: " + str(i) + ' ' + str(j) + '\n')
        filePtr.write(str(accuracy) + '\n\n')

        fig, ax = plt.subplots() # figsize=(6,10)
        plt.title('Confusion matrix with features: ' + features[i] + ' & ' + features[j])
        ax = sns.heatmap(confDF, annot = True, square=True, linewidth=3)
        fig.savefig('Heatmaps/heatmap ' + str(i) + ' ' + str(j) + ' .png')

for i in range(0, 4):
    for j in range(i+1, 4):
        for k in range(j+1, 4):
            currentDF = pd.DataFrame()
            currentDF.insert(0, features[i], irisColumns[i].to_list(), True)
            currentDF.insert(1, features[j], irisColumns[j].to_list(), True)
            currentDF.insert(2, features[k], irisColumns[k].to_list(), True)
            #print(currentDF)
            model.fit(currentDF, irisLabels)
            accuracy = model.score(currentDF, irisLabels)
            predicted = model.predict(currentDF)
            confusion = confusion_matrix(irisLabels, predicted)
            confDF = pd.DataFrame(confusion)
            confDF.columns = categories
            confDF.index = categories

            filePtr.write("Features: " + str(i) + ' ' + str(j) + ' ' + str(k) + '\n')
            filePtr.write(str(accuracy) + '\n\n')

            fig, ax = plt.subplots() # figsize=(6,10) 
            plt.title('Confusion matrix with features: ' + features[i] + ', ' + features[j] + ', and ' + features[k])
            ax = sns.heatmap(confDF, annot = True, square=True, linewidth=3)
            fig.savefig('Heatmaps/heatmap ' + str(i) + ' ' + str(j) + ' ' + str(k) + ' .png')