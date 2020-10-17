import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import *
from sklearn.naive_bayes import *

iris = pd.read_csv('Iris.csv', header = None)
iris.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'category']
categories = ['Setosa', 'Versicolor', 'Virginica']
irisLabels = iris['category'] 
iris = iris.drop(labels='category', axis=1)
print(irisLabels, end='\n\n\n')
print(iris) 

model = GaussianNB()
model.fit(iris, irisLabels)
accuracy = model.score(iris, irisLabels)
print(accuracy)
predicted = model.predict(iris)
confusion = confusion_matrix(irisLabels, predicted)
confDF = pd.DataFrame(confusion)
confDF.columns = categories
confDF.index = categories

fig, ax = plt.subplots() # figsize=(6,10)
#ax = 
#plt.figure()
ax = sns.heatmap(confDF, annot = True, square=True, linewidth=3)
#bottom, top = ax.get_ylim()
#print(bottom, top)
#ax.set_ylim(bottom, top)
#plt.show()
#ax.set_ylim(5, 5)
fig.savefig('Heatmaps/heatmap 0 1 2 3.png')