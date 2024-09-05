import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import HuberRegressor 

# importing dataset : Health experiment based on age and physical test scores
# Passing test is '1' while not passing is '0' in the test_result column

data = pd.read_csv('./Datasets/experiment.csv')

# Some Starters 
# print(data.head())
# print(data.describe().transpose())
# print(data['test_result'].value_counts()) # 3000 people pass the test while 2000 don't

# Visualization
# sns.countplot(data=data , x='test_result' , hue='test_result')
# sns.boxplot(data=data , x='test_result' , y='age' , hue='test_result')
# sns.boxplot(data=data , x='test_result' , y='physical_score' , hue='test_result')
# sns.scatterplot(data=data , x='age' , y='physical_score' , hue='test_result')
# sns.pairplot(data=data , hue='test_result')
# sns.heatmap(data=data.corr() , annot=True)

# 3-D plot
# from mpl_toolkits.mplot3d import Axes3D
# fig=plt.figure()
# ax = fig.add_subplot(111 , projection='3d')
# ax.scatter(data['age'] , data['physical_score'] , c=data['test_result'])
#plt.show()


# Seprating Label and Features
X = data.drop('test_result' , axis=1)
y = data['test_result']

# Spliting and Scaling the features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train , X_test , y_train , y_test = train_test_split(X , y , random_state=42 , test_size=0.1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Creating Model 
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train , y_train)

print(model.coef_)

y_pred = model.predict(X_test) # Prediction in '0' or '1'
y_pred = model.predict_proba(X_test) # returns probabilities of point belonging to a specific class
y_pred = model.predict_log_proba(X_test) # returns log() of probability
