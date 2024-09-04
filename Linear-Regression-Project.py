# A Linear-Regression model to predict the SalePrice of a house
# based on the data "AMES_FINAL_DF.csv"

from tkinter import Grid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("./Datasets/AMES_FINAL_DF.csv")

# Seprating Labels
X = data.drop('SalePrice' , axis=1)
y = data['SalePrice']

# Defining Train-Test Split
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y , random_state=42 , test_size=0.1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Creating the model 
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

MODEL = ElasticNet(max_iter=100000)

prams_grid = {
    'l1_ratio': [0.1 , 0.5 , 0.77 , 0.95 , 1],
    'alpha':[0.99 , 1 , 2 , 3 , 5]
}

Grid_Search_Model = GridSearchCV(estimator=MODEL , param_grid=prams_grid , cv=5 , scoring='neg_root_mean_squared_error')
Grid_Search_Model.fit(X_train , y_train)

# print(Grid_Search_Model.best_params_) {'alpha': 0.99, 'l1_ratio': 0.95}

# y_pred = Grid_Search_Model.predict(X_test)

# Calculating Performance

from sklearn.metrics import mean_absolute_error , mean_squared_error
# print(mean_absolute_error(y_test , y_pred )) # Mean Absolute Error : 13851.04299852958
# print(np.sqrt(mean_squared_error(y_test , y_pred ))) # Root Mean Squared Error :  21964.480975143433