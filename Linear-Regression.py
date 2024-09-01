import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

# a simple linear regression (using ols) can be coded as follows :

# data for Linear regression
data = pd.read_csv("./Datasets/Advertising.csv")

# Generating Sum Expense of Advertising [ feature engineering ]
data['Total Expense'] = data['TV'] + data['radio'] + data['newspaper']
# print(data.head())

# Visualing data as a Scatterplot 
sns.scatterplot(x='Total Expense' , y='sales' , data=data)
sns.regplot(x='Total Expense' , y='sales' , data=data)
# plt.show()

# Feature Matrix and Label
X = data['Total Expense']
y = data['sales']

# y = B1x + B0
# print(np.polyfit(X,y , deg=1)) # gives coeffs B1 and B0 for the equation above

# a simple ordinary least squares for regression
test_expense = np.linspace(0 , 500 , 100)
#       y                        B1             x                B0
prediction = np.polyfit(X,y , deg=1)[0] * test_expense + np.polyfit(X,y , deg=1)[1]

# plotting predictions vs actual spend
plt.plot(test_expense , prediction , color='red')
# plt.show()

# testing 
spend = 359
predicted_sales =  np.polyfit(X,y , deg=1)[0] * spend + np.polyfit(X,y , deg=1)[1]
# print(predicted_sales)

#-----------------------------------------------------------------------------------------