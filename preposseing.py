# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 20:08:37 2020

@author: hana
"""
import numpy as np
import pandas as pd

df = pd.read_csv(r'C:/Users/hana/Desktop/مقاله/DeepMal-master/H. sapiens/2/combine (EAAC,PKA,POSITION,MyWeight)/PPEWE_combin_train.csv')
df.drop('Unnamed: 0', inplace=True, axis=1)


        
w=df.describe()



from sklearn import preprocessing

#z-score
std_scale = preprocessing.StandardScaler().fit(df)
df_std = std_scale.transform(df)


#min-max
minmax_scale = preprocessing.MinMaxScaler().fit(df)
df_minmax = minmax_scale.transform(df)

data1=np.matrix(df_minmax[:])[:,:]
data_=pd.DataFrame(data=data1)


q=data_.std()

"""drup column"""

for i in range(574):
    if(q[i]==0):
        #a=str(i)
        data_.drop(i,inplace=True, axis=1)

data1=np.matrix(data_[:])[:,:]
data1_=pd.DataFrame(data=data1)


data1_.to_csv('C:/Users/hana/Desktop/hananeh/H/(EAAC,PKA,POSITION,MyWeight,EGAAC)_MINMAX_combin_train.csv')
#data1_.to_csv('C:/Users/hana/Desktop/hananeh/H/(EAAC,PKA,POSITION,MyWeight,EGAAC)_fscor_combin_train.csv')

#df_minmax.to_csv('BDEEKT-min-max_combin_train.csv')




