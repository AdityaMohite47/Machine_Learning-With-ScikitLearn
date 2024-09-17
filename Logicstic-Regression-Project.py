import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings

# This database contains 14 physical attributes based on physical testing of a patient. 
# Blood samples are taken and the patient also conducts a brief exercise test. 
# The "goal" field refers to the presence of heart disease in the patient. It is integer (0 for no presence, 1 for presence). 

# Attribute Information:
# * age
# * sex
# * chest pain type (4 values)
# * resting blood pressure
# * serum cholestoral in mg/dl
# * fasting blood sugar > 120 mg/dl
# * resting electrocardiographic results (values 0,1,2)
# * maximum heart rate achieved
# * exercise induced angina
# * oldpeak = ST depression induced by exercise relative to rest
# * the slope of the peak exercise ST segment
# * number of major vessels (0-3) colored by flourosopy
# * thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
# * target:0 for no presence of heart disease, 1 for presence of heart disease

# Original Source: https://archive.ics.uci.edu/ml/datasets/Heart+Disease

# Creators:
# Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
# University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
# University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
# V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.

# ---------------------------------------------------------------------------------------------------------------------------------------

data = pd.read_csv('./Datasets/heart.csv')

# print(data.info())
# print(data.sample(10))

# checking if dataset has any feature value missing
# print(data.isna().sum())

# ---------------------------------------------------------------------------------------------------------------------------------------

# Visualization on data

# Count per category
# sns.countplot(x='target' , data=data , hue='target')

# Pair plot on ['age','trestbps', 'chol','thalach','target']
# sns.pairplot(data=data , vars=['age','trestbps', 'chol','thalach','target'] , hue='target')

# Heatmap on Correlation of data
# sns.heatmap(data=data.corr() , annot=True)
# plt.show()

# ---------------------------------------------------------------------------------------------------------------------------------------

# Prepraing data and making model

X = data.drop('target' , axis=1)
y = data['target']

from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

X_train , X_test , y_train , y_test = train_test_split(X , y , random_state=42 , test_size=0.3)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

base_model = OneVsRestClassifier(estimator=LogisticRegression(max_iter=5000 , solver='saga'))

# supressing the warings , as the 'l1_ratio' is only used when penalty is 'elastic'.
warnings.filterwarnings("ignore", message="l1_ratio parameter is only used when penalty is 'elasticnet'")

params_grid = {
    'estimator__penalty': ['l1' ,'l2' , 'elasticnet'],
    'estimator__l1_ratio': np.linspace(0, 1, 20),
    'estimator__C': np.linspace(0.01, 2, 20)  
}

MODEL = GridSearchCV(estimator=base_model , param_grid=params_grid , cv=5 , )
MODEL.fit(X_train , y_train)

# print(MODEL.best_params_) 

# {
# 'estimator__C': np.float64(0.11473684210526315), 
#  'estimator__l1_ratio': np.float64(0.0), 
#  'estimator__penalty': 'l2'
# }

# ---------------------------------------------------------------------------------------------------------------------------------------

# Predictions on test set

y_pred = MODEL.predict(X_test)

from sklearn.metrics import accuracy_score , precision_score , recall_score , confusion_matrix

print(accuracy_score(y_test , y_pred))
# 0.824

print(precision_score(y_test , y_pred))
print(recall_score(y_test , y_pred ))
# precision and recall both 0.84

print(confusion_matrix(y_test , y_pred)) # all-over 16 wrong predictions...
# [[33  8]
#  [ 8 42]]

# ---------------------------------------------------------------------------------------------------------------------------------------