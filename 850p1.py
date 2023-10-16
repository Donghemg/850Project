# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 16:06:12 2023

@author: 15127
"""

# #1, split,stratify 
import pandas as pd

df = pd.read_csv("Project 1 Data.csv")

# #2
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
#data normalize 



df = df.dropna()
df = df.reset_index(drop=True) 
missing_values = df.isna().any()

#data visualization 

# Create a line plot
plo1=plt.plot(df["XX"],df["Step"])
# Set labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('X')
# Show the plot
plt.show(plo1)

plo2=plt.plot(df["YY"],df["Step"])
# Set labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Y')
# Show the plot
plt.show(plo2)
 
plo3=plt.plot(df["ZZ"],df["Step"])
# Set labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Z')
# Show the plot
plt.show(plo3)
 # strat
df["Step_cat"] = pd.cut(df["Step"],bins=[0.99, 5, 9, 13.1],labels=[1,2,3])
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df["Step_cat"]):
    strat_train_set = df.loc[train_index].reset_index(drop=True)
    strat_test_set = df.loc[test_index].reset_index(drop=True)
strat_train_set = strat_train_set.drop(columns=["Step_cat"], axis = 1)
strat_test_set = strat_test_set.drop(columns=["Step_cat"], axis = 1)



#Training

train_y = strat_train_set['Step']
df_X = strat_train_set.drop(columns = ['Step'])




my_scaler = StandardScaler()
my_scaler.fit(df_X)
scaled_data = my_scaler.transform(df_X)
scaled_data_df = pd.DataFrame(scaled_data, columns=df_X.columns)
train_X = scaled_data_df

test_y = strat_test_set['Step']
df_test_X = strat_test_set.drop(columns = ["Step"])
scaled_data_test = my_scaler.transform(df_test_X)
scaled_data_test_df = pd.DataFrame(scaled_data_test, columns=df_test_X.columns)
test_X = scaled_data_test_df


columns_list = train_X.columns.tolist()
new_order = ['XX',
  'YY',
  'ZZ']
# train_X = train_X[new_order]
# test_X = test_X[new_order]
# train_X=train_X.join

corr_matrix = train_X.corr()
sns.heatmap(np.abs(corr_matrix))
corr1 = np.corrcoef(train_X['XX'], train_y)
#print(corr1[0,1])
corr2 = np.corrcoef(train_X['YY'], train_y)
#print(corr2[0,1])
corr3 = np.corrcoef(train_X['ZZ'], train_y)
#print(corr3[0,1])



plt.figure()
sns.heatmap(np.abs(corr_matrix))

# 2.4
# linear regression
from sklearn.linear_model import LinearRegression
model1 = LinearRegression()
model1.fit(train_X, train_y)

from sklearn.ensemble import RandomForestClassifier  # for classification tasks
from sklearn.ensemble import RandomForestRegressor  # for regression tasks

# Create a Random Forest model
model2 = RandomForestClassifier(n_estimators=100)  # Specify the number of trees

# Fit the model to your training data
model2.fit(train_X, train_y)

# # Make predictions
predics = model2.predict(test_X)

# # Evaluate the model
acc_RT = model2.score(test_X, test_y)
acc_LR = model1.score(test_X, test_y)

#2.5
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(predics, test_y)
print(cm)

#2.6
import joblib

# Save a scikit-learn model to a file
joblib.dump(model1, 'model1.pkl')
joblib.dump(model2, 'model2.pkl')


#[9.375,3.0625,1.51], [6.995,5.125,0.3875], [0,3.0625,1.93], [9.4,3,1.8], [9.4,3,1.3]
arr1 = np.array([9.375, 3.0625, 1.51])

arr1p = model2.predict(arr1.reshape(1,-1))
print (arr1p)

arr2 = np.array([6.995,5.125,0.3875])
arr2p = model2.predict(arr2.reshape(1,-1))
print (arr2p)

arr3 = np.array([0,3.0625,1.93])
arr3p = model2.predict(arr3.reshape(1,-1))
print (arr3p)

arr4 = np.array([9.4,3,1.8])
arr4p = model2.predict(arr4.reshape(1,-1))
print (arr4p)

arr5 = np.array([9.4,3,1.3])
arr5p = model2.predict(arr5.reshape(1,-1))
print (arr5p)


arr1p1 = model1.predict(arr1.reshape(1,-1))
arr1p2 = model1.predict(arr2.reshape(1,-1))
arr1p3 = model1.predict(arr3.reshape(1,-1))
arr1p4 = model1.predict(arr4.reshape(1,-1))
arr1p5 = model1.predict(arr5.reshape(1,-1))
print(arr1p1,arr1p2,arr1p3,arr1p4,arr1p5)













