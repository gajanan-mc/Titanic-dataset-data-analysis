# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 09:56:34 2023

@author: DELL
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df  = pd.read_csv(r'C:\Users\DELL\Downloads\Monty\26th\project - data preprocessing\train.csv')


df.drop('Cabin',axis=1,inplace=True)
df.info()

df.Age.isnull().sum()
df.columns

X = df.drop('Survived', axis=1)
Y = df.Survived

X
sns.boxplot(df['Age'])
# as we have outliers in Age feature so we are imputing it with median

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer = imputer.fit(X[['Age']])
X['Age'] = imputer.transform(X[['Age']])


df.groupby('Embarked')['Survived'].value_counts()

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.30,random_state=10)


















