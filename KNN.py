# Developing a simple KNN model with k = 1 ! 

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

 # 2 Gene Expression features that identify wheter a person has Cancer or not
data = pd.read_csv('./Datasets/gene_expression.csv')
# print(data.head())
# print(data.info())

# sns.scatterplot(data=data , x='Gene One' , y='Gene Two' , hue='Cancer Present' , alpha=0.5)
# plt.show()

X = data.drop('Cancer Present' , axis=1)
y = data['Cancer Present']

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.fit_transform(X_test)

# from sklearn.neighbors import KNeighborsClassifier
# model = KNeighborsClassifier(n_neighbors=1)
# model.fit(X_train , y_train)

# y_pred = model.predict(X_test)

# # print(confusion_matrix(y_test , y_pred))
# print(classification_report(y_test , y_pred))

#---------------------------------------------------------------
# choosing optimal 'k' with GridSearchCV and learning pipeline 

# making pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

scaler = StandardScaler()
model = KNeighborsClassifier()

operations = [ ('scaler',scaler) , ('model' , model) ]

pipe = Pipeline(steps=operations)

params_grid = {
        'model__n_neighbors': [x for x in range(1 , 20) ]        
}

GridSearchKnn = GridSearchCV(pipe , param_grid=params_grid , cv=5 , scoring='accuracy')
GridSearchKnn.fit(X_train , y_train)

print(GridSearchKnn.best_params_)

y_pred = GridSearchKnn.predict(X_test)
print(confusion_matrix(y_test , y_pred))
print(classification_report(y_test , y_pred))
