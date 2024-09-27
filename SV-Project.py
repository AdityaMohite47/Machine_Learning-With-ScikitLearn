## Project Goals

# A distribution company that was recently a victim of fraud has completed an audit of various samples of wine through the use of 
# chemical analysis on samples. The distribution company specializes in exporting extremely high quality, expensive wines, but was 
# defrauded by a supplier who was attempting to pass off cheap, low quality wine as higher grade wine. The distribution company has 
# hired you to attempt to create a machine learning model that can help detect low quality (a.k.a "fraud") wine samples. 
# They want to know if it is even possible to detect such a difference.


# Data Source: *P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties.
# In Decision Support Systems, Elsevier, 47(4):547-553, 2009.*


# **TASK: Your overall goal is to use the wine dataset shown below to develop a machine learning model that attempts to predict if 
# a wine is "Legit" or "Fraud" based on various chemical features.


from random import sample
from tkinter import Grid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics

data = pd.read_csv('./Datasets/wine_fraud.csv')
# print(data.sample(10))
# print(data.columns)

# Comparing record count of legit and fraud
# sns.countplot(data=data , x='quality')

# # Comparing record count of legit and fraud wine types
# sns.countplot(data=data , x='quality' , hue='type')


# Correlation of each feature with traget

data['quality'] = data['quality'].map({'Legit':0 , 'Fraud':1})
data['type'] = data['type'].map({'red':0 , 'white':1})
# print(data.corr()['quality'].sort_values())

# sns.heatmap(data=data.corr() , annot=True)
# # plt.show()

#--------------------------------------------------------------------------------------------------------------------------------

import sklearn
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import warnings # for suppressing unecessary warnings

warnings.filterwarnings("ignore", message="degree parameter is only used when kernel is 'poly'")


X = data.drop('quality' , axis=1 )
y = data['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC()
params_grid = {
    'kernel': ['linear', 'poly', 'rbf'],
    'C':[0.1 , 0.2 , 0.3 , 0.4 , 0.5],
    'gamma': [0.01 , 0.02 , 0.03 , 0.04 , 0.05],
    'degree': [x for x in range(20)]
}

# The GridSearch had taken lot of time...
Grid_model = GridSearchCV(estimator=model , param_grid=params_grid , cv=10)
Grid_model.fit(X_train , y_train)


# print(Grid_model.best_estimator_) # SVC(C=0.1, degree=11, gamma=0.03, kernel='poly')

#-----------------------------------------------------------------------------------------------------------------------------
# Testing model with test set

from sklearn.metrics import classification_report , confusion_matrix
from imblearn.over_sampling import SMOTE

y_pred = Grid_model.predict(X_test)

# Confusion Matrix
print(confusion_matrix(y_test , y_pred))
# [[1870    2]
#  [  76    2]]

# Classification Report
print(classification_report(y_test , y_pred))
#               precision    recall  f1-score   support

#            0       0.96      1.00      0.98      1872
#            1       0.50      0.03      0.05        78

#     accuracy                           0.96      1950
#    macro avg       0.73      0.51      0.51      1950
# weighted avg       0.94      0.96      0.94      1950

