# -*- coding: utf-8 -*-
"""
Created on Mon May  9 20:30:44 2022

@author: User
"""

import pickle

#Loading
import pandas as pd
data=pd.read_csv("data.csv")

#filling missing values
data=data.dropna()
data=data.drop([195073, 195282,273577])

ind1=data[data['Age']== ' '].index
data['Age'][ind1]='45-54'

ind2=data[data['Sex']== ' '].index
data['Sex'][ind2]='Female'

ind3=data[data['Condition']== ' '].index
data=data.drop(ind3)

#feature engineering
data=data.drop(['Reviews','DrugId','Date'],axis=1)

#selecting data
df=data.groupby('Drug')
df1=df.get_group('lisinopril solution')

df1=df1.drop(['Drug','UsefulCount','Sides'],axis=1)

drug_data=df1.copy()
drug_data=pd.get_dummies(df1)

x=drug_data.drop(['Effectiveness'],axis=1)
y=drug_data['Effectiveness']

y[y==1]='Extreme side effects'
y[y==2]='Severe Side effects'
y[y==3]='Moderate Side Effects'
y[y==4]='Mild Side Effects'
y[y==5]='No Side Effects'

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

from sklearn.ensemble import GradientBoostingClassifier
gb=GradientBoostingClassifier()
gb.fit(X_train,y_train)

#Saving the model to disk
pickle.dump(gb,open('quality.pkl','wb'),2 )

cat=x
index_dict = dict(zip(cat.columns,range(cat.shape[1])))
pickle.dump(index_dict,open('cat','wb'),2 )
