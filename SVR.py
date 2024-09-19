# Support Vector Regression

from tkinter import Grid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from svm_margin_plot import plot_svm_boundary

# Dataset about Concrete Slump test
data = pd.read_csv('./Datasets/cement_slump.csv')
# print(data.sample(10))
# print(data.info())

# Correlation 
# sns.heatmap(data=data.corr() , annot=True)
# plt.show()
# print(data.corr()['Compressive Strength (28-day)(Mpa)'].sort_values())

# Setting up features and executing all steps in a pipline
X = data.drop('Compressive Strength (28-day)(Mpa)' , axis=1)
y = data['Compressive Strength (28-day)(Mpa)']

from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
pipe = Pipeline(steps=[ ('scaler' , StandardScaler()) , ('model' , SVR())])
params_grid = {
    'model__kernel':['linear' , 'rbf'],
    'model__gamma':[0.001 , 0.1 , 0.2 , 0.3 , 0.4 , 0.5], # how much influence of a datapoint
    'model__C':[0.1 , 0.2 , 0.3 , 0.4 , 0.5], # Regularization Strength
    'model__epsilon':[0.1 , 0.2 , 0.3 , 0.4 , 0.5], # how much error is allowed
}

GridSearch_SVR = GridSearchCV(estimator=pipe , param_grid=params_grid , scoring='neg_root_mean_squared_error')
GridSearch_SVR.fit(X_train , y_train)

# print(GridSearch_SVR.best_params_) # {'model__C': 0.5, 'model__epsilon': 0.4, 'model__gamma': 0.1, 'model__kernel': 'linear'}
# print(GridSearch_SVR.best_score_) # -3.2916905332437167

y_pred = GridSearch_SVR.predict(X_test)

# Evaluation
from sklearn.metrics import mean_absolute_error , mean_squared_error
print(mean_absolute_error(y_test , y_pred)) # 2.2026986227738417
print(np.sqrt(mean_squared_error(y_test , y_pred))) # 2.81149658845758
