## The Sonar Data -> # Detecting a Rock or a Mine

# Sonar (sound navigation ranging) is a technique that uses sound propagation (usually underwater, as in submarine navigation) 
# to navigate, communicate with or detect objects on or under the surface of the water, such as other vessels.

# Data consists of 60 features ( different frequencies tested on object) and categorical target variable 
# 'Label' ( 'R' for Rock and 'M' for a Mine) 

from tkinter import Grid
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# Importing and Analyzing the data...
data = pd.read_csv('./Datasets/sonar.all-data.csv')
# print(data.info())
# print(data.sample(10))

data['Label'] = data['Label'].map({'R':0 , 'M':1})

# top 5 features correlated to target
# print(data.corr()['Label'].sort_values()[-6:-1])

#-----------------------------------------------------------------------------------------------------------------------------

# Data design , model traning and testing...

from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix , classification_report

X = data.drop('Label' , axis=1)
y = data['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


pipe = Pipeline(steps=[('scaler',StandardScaler()) , ('model' , KNeighborsClassifier())])

params_grid = {
    'model__n_neighbors': [x for x in range(1 , 31)]
}

GridSearchModel = GridSearchCV(estimator=pipe , scoring='accuracy' , cv=5 , param_grid=params_grid)
GridSearchModel.fit(X_train , y_train)
print(GridSearchModel.best_params_)

y_pred = GridSearchModel.predict(X_test)

print(confusion_matrix(y_test , y_pred))
print(classification_report(y_test , y_pred))