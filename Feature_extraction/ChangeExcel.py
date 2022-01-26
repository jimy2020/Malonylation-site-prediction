# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:57:33 2020

@author: Administrator
"""
#import shutil
import pandas as pd
import numpy as np
dbname='E'
t='train'
#FeatureName='BLOSUM62' # BLOSUM62 DDE EAAC EGAAC KNN Hash LDA1
whichfeature=[2]
listFeature=['Feature_extraction/BLOSUM62/BLOSUM62_'+dbname+'_'+t+'.csv','Feature_extraction//DDE/DDE_'+dbname+'_'+t+'.csv','Feature_extraction/EAAC/EAAC_'+dbname+'_'+t+'.csv','Feature_extraction/EGAAC/EGAAC_'+dbname+'_'+t+'.csv','Feature_extraction/KNN/KNN_'+dbname+'_'+t+'.csv','Feature_extraction/Hash/Hash_'+dbname+'_'+t+'.csv','Feature_extraction/LDA1/LDA1_'+dbname+'_'+t+'.csv','Feature_extraction/LDA1/LDA2_'+dbname+'_'+t+'.csv','Feature_extraction/weight/weight_'+dbname+'_'+t+'.csv']

all_data = pd.DataFrame()
a=[]
i=0
for f in listFeature:
    if whichfeature.count(i)>0 :
        a.append(pd.read_csv(f))
    i=i+1
concatenated=pd.concat(a,axis=1)  
concatenated.drop('Unnamed: 0', inplace=True,axis=1)  
concatenated.to_csv('Classifier/All_'+dbname+'_'+t+'.csv')
#all_data = all_data.append(df,ignore_index=True)
#all_data.to_csv('All_'+dbname+'_'+t+'.csv')

t='test'
listFeature=['Feature_extraction/BLOSUM62/BLOSUM62_'+dbname+'_'+t+'.csv','Feature_extraction//DDE/DDE_'+dbname+'_'+t+'.csv','Feature_extraction/EAAC/EAAC_'+dbname+'_'+t+'.csv','Feature_extraction/EGAAC/EGAAC_'+dbname+'_'+t+'.csv','Feature_extraction/KNN/KNN_'+dbname+'_'+t+'.csv','Feature_extraction/Hash/Hash_'+dbname+'_'+t+'.csv','Feature_extraction/LDA1/LDA1_'+dbname+'_'+t+'.csv','Feature_extraction/LDA1/LDA2_'+dbname+'_'+t+'.csv','Feature_extraction/weight/weight_'+dbname+'_'+t+'.csv']
all_data = pd.DataFrame()
a=[]
i=0
for f in listFeature:
    if whichfeature.count(i)>0 :
        a.append(pd.read_csv(f))
    i=i+1
concatenated=pd.concat(a,axis=1)  
concatenated.drop('Unnamed: 0', inplace=True,axis=1)  
concatenated.to_csv('Classifier/All_'+dbname+'_'+t+'.csv')
