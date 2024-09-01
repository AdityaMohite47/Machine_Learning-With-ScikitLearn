import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

df = pd.read_csv('./Datasets/penguins_size.csv')

# print(df.head())
# print(df.info())
# print(df.corr()['species'])
# print(df.isnull().sum())

# dropping null values
df = df.dropna()

# print(df[df['species']=='Gentoo'].groupby('sex').describe().T)
df.at[336 , 'sex'] = 'FEMALE'

# sns.pairplot(df,hue='species')
# plt.show()

X = pd.get_dummies(df.drop('species' , axis=1) , dtype=int , drop_first=True)
# print(X[:5])
y = df['species']

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier , plot_tree
from sklearn.metrics import classification_report , confusion_matrix

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train , y_train)

base_pred = model.predict(X_test)
# print(classification_report(y_test , base_pred))
# print(confusion_matrix(y_test , base_pred))
# print(pd.DataFrame(index=X.columns , data=model.feature_importances_ , columns=['Feature Importances']).sort_values('Feature Importances'))
fig = plt.figure(figsize=[12.0 , 12.0] , dpi=100)
plot_tree(model)
plt.show()
