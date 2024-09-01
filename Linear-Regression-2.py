import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data Prepration   
data = pd.read_csv("./Datasets/Advertising.csv")

X = data.drop('sales' , axis=1)
y = data['sales']


# ------------------------------------------------------------------

# Polynomial features through sklearn...

from sklearn.preprocessing import PolynomialFeatures

poly_converter = PolynomialFeatures(degree=2 , include_bias=False)
X = poly_converter.fit_transform(X)

# ------------------------------------------------------------------

# Making a Train-Test split...

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y ,test_size=0.3 , random_state=42)

# ------------------------------------------------------------------

# Scaling the features of the data with sklearn...

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------------------------------------------

# Some Notes :

# constant in the equation of any algorithm which is a tunable hyper-parameters
# is refered as alpha in sklearn

# The Cross-Validation uses the scoring param which is used to 
# find the best hyper-param value for a model And the higher the value of the 
# test the better the model is.
# This is to maintain uniformity across all scoring metrics.

# A higher accuracy is better but a higher RMSE is worse. Sklearn fixes it 
# by using -ve RMSE as its scorer metric

# ------------------------------------------------------------------

# l2 Eegularization / Ridge Regression

from sklearn.linear_model import Ridge , RidgeCV
from sklearn.metrics import mean_squared_error

ridge_model = Ridge(alpha=10)
ridge_model.fit(X_train , y_train)

# print(ridge_model.coef_)

test_pred = ridge_model.predict(X_test)

# RMSE of the base model
# print(np.sqrt(mean_squared_error(y_test , test_pred)))

# Performing cross-validation for the best alpha

ridge_cv_model = RidgeCV(alphas=(0.1 , 0.4 , 0.9 , 1.0 , 10.0) , scoring='neg_root_mean_squared_error')
ridge_cv_model.fit(X_train , y_train)

# print(ridge_cv_model.alpha_) # 0.1 was the best performing alpha   

test_pred = ridge_cv_model.predict(X_test)
# RMSE of the cv model
# print(np.sqrt(mean_squared_error(y_test , test_pred)))

# ------------------------------------------------------------------

# L1 Regularization / Lasso Regression

from sklearn.linear_model import LassoCV

lasso_cv_model = LassoCV(eps=0.001 , n_alphas=100 , cv=8)

# same as np.linspace eps/epsilon acts as starting point and n_alphas as
# no. of values to discover further

lasso_cv_model.fit(X_train , y_train)

test_pred = lasso_cv_model.predict(X_test)

# RMSE of the lasso cv model
# print(np.sqrt(mean_squared_error(y_test , test_pred)))

# ------------------------------------------------------------------

# Elastic Net Regularization : L1 + L2

from sklearn.linear_model import ElasticNetCV

# refers to                         alpha-ratio of l1:l2                  lambda values to consider
elastic_cv_model = ElasticNetCV(l1_ratio=[.1 , .3 , .5 , .7 , .9 , 1] , eps=0.001 , n_alphas=100)

elastic_cv_model.fit(X_train , y_train)

# print(elastic_cv_model.l1_ratio_) # 0.9 
# print(elastic_cv_model.alpha_)

test_pred = lasso_cv_model.predict(X_test)

# RMSE of the elastic cv model
print(np.sqrt(mean_squared_error(y_test , test_pred)))

# Note : for small datasets elasticnet_model decides to
# stay with lasso only i.e alpha ratio doesn't matter for 
# small datasets in elasticnet

# ------------------------------------------------------------------