import pandas as pd
import numpy as np
import seaborn as sns
import pickle

data = pd.read_csv("data_cardiovascular_risk.csv")

#data = data.apply(pd.to_numeric,errors = 'coerce').fillna(0)

data.head()

data.describe()

data.isnull().sum()

sns.kdeplot(data.education,fill = True)

data['education'].fillna(data['education'].mean(), inplace = True)

sns.kdeplot(data.cigsPerDay, fill = True)

# Data of ciggarates per day looked to be skewed hence we replace the null value with median

data['cigsPerDay'].fillna(data['cigsPerDay'].median() , inplace = True)

sns.kdeplot(data.BPMeds , fill = True)

# data of BP medicine appear to be positively skewed hence replacing the null values with median


data['BPMeds'].fillna(data['BPMeds'].median(), inplace = True)

sns.kdeplot(data.totChol , fill = True)

# Positively skewed hence filling null values with median
data['totChol'].fillna(data['totChol'].median() , inplace = True)

sns.kdeplot(data.BMI , fill = True)

# data is positively skewed hence replacing null values with median
data['BMI'].fillna(data['BMI'].median() , inplace = True)

sns.kdeplot(data.glucose , fill = True)

# data is positively skewed hence replacing null values with median

data['glucose'].fillna(data['glucose'].median(), inplace = True)

data['heartRate'].fillna(method = 'bfill' , inplace = True)

data.isnull().sum()

data['sex'].replace(['F','M'], [0,1] , inplace = True)

data['sex'].unique()

data['is_smoking'].unique()

data['is_smoking'].replace(['YES', 'NO'] , [1,0], inplace = True)

data['is_smoking'].unique()
data.drop('id',axis=1,inplace=True)

x = data.iloc[:,0:-1].values
y = data.iloc[:,15].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2 , random_state =0)

from sklearn import svm
model = svm.SVC(kernel = 'linear')
model.fit(x_train , y_train)
pickle.dump(model,open('model.pkl','wb'))
