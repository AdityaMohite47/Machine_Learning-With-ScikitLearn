import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from svm_margin_plot import plot_svm_boundary


# Dataset is about the infected mouse been given doses to recover
# and after 2 weeks the mouse's codition is checked ( Traget variable )
data = pd.read_csv('./Datasets/mouse_viral_study.csv')
# print(data.sample(10))

# Perfect seperated classes 
# sns.scatterplot(x='Med_1_mL' , y='Med_2_mL' , data=data , hue='Virus Present')
# plt.show()

# Making model with Support Vector Classifier to visualize hyperplane
from sklearn.svm import SVC
X = data.drop('Virus Present' , axis=1)
y = data['Virus Present']

model = SVC(kernel='linear' , C=1000)
model.fit(X , y)

plot_svm_boundary(model , X , y) 
# This 3-rd party function helps us to visualize hyperplane made by the kernel 
# It also highlights the support vectors and margins/soft-margins...