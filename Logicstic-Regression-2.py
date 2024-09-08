# Multiclass Logicstic Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.multiclass import OneVsRestClassifier

data = pd.read_csv('./Datasets/iris.csv')


# Iris dataset has data of flowers categorized in 3 types
# print(data['species'].unique()) # ['setosa' 'versicolor' 'virginica']

# Visualizing balance between classes

# sns.countplot(x='species'  , data=data)
# sns.scatterplot(data=data , x='petal_length' , y='petal_width' , hue='species')
# sns.pairplot(data=data , hue='species')
# sns.heatmap(data=data.corr() , annot=True)
# plt.show()

# Mapping for visualization purposes
# data['species'] = data['species'].map({'setosa':1 , 'versicolor':2 , 'virginica':3})

# Not hot-encoding the classes is fine with sklearn

# ---------------------------------------------------------------------------------------------------------

X = data.drop('species' , axis=1)
y = data['species']

# Spliting and Scanning the data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=42)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# ---------------------------------------------------------------------------------------------------------

# Making on Model
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import GridSearchCV 

# Applying OVR - One vs Rest test for multi-class Logicstic Regression
OVR_MODEL  = OneVsRestClassifier(estimator=LogisticRegression(solver='saga' , max_iter=5000))

# supressing the warings , as the 'l1_ratio' is only used when penalty is 'elastic'.
warnings.filterwarnings("ignore", message="l1_ratio parameter is only used when penalty is 'elasticnet'")

params_grid = {
    'estimator__penalty': ['l1' ,'l2' , 'elasticnet'],
    'estimator__l1_ratio': np.linspace(0, 1, 20),
    'estimator__C': np.linspace(0.01, 2, 20)  
} 

GridSearchModel = GridSearchCV(estimator=OVR_MODEL , param_grid=params_grid)
GridSearchModel.fit(X_train , y_train)

# Best params
# print(GridSearchModel.best_params_) 
# ->
    # {'estimator__C': np.float64(1.4763157894736842), 
    #'estimator__l1_ratio': np.float64(1.0),
    #'estimator__penalty': 'elasticnet'}

# ---------------------------------------------------------------------------------------------------------

# Predictions

from sklearn.metrics import accuracy_score , precision_score , recall_score , confusion_matrix

y_pred = GridSearchModel.predict(X_test)


print(accuracy_score(y_test , y_pred))
print(precision_score(y_test , y_pred , average='macro'))
print(recall_score(y_test , y_pred , average='macro'))
# All above tests results in 1.0 [ 100 % ]

print(confusion_matrix(y_test , y_pred))
# [[15  0  0]
#  [ 0 11  0]
#  [ 0  0 12]]

# ---------------------------------------------------------------------------------------------------------

# Test 
test_data = pd.read_csv('./Datasets/iris_test.csv')

predictions = GridSearchModel.predict(test_data.drop('species' , axis=1))

print(accuracy_score(test_data['species'] , predictions)) # 0.9
print(precision_score(test_data['species'] , predictions , average='macro')) # 0.91
print(recall_score(test_data['species'] , predictions , average='macro')) # 0.88
print(confusion_matrix(test_data['species'] , predictions))
# [[4 0 0]
#  [0 2 1]
#  [0 0 3]]

# ---------------------------------------------------------------------------------------------------------