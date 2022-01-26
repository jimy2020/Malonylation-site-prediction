# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 12:16:20 2020

@author: jamshid
"""

import pandas as pd

B=pd.read_csv(r'BLOSUM62\BLOSUM62_E_train.csv')
D=pd.read_csv(r'DDE\DDE_E_train.csv')
E=pd.read_csv(r'EGAAC\EGAAC_E_train.csv')
E1=pd.read_csv(r'EAAC\EAAC_E_train.csv')
k=pd.read_csv(r'KNN\KNN_E_train.csv')

concatenated = pd.concat([D, B,E,E1,k], axis=1)
concatenated.drop('Unnamed: 0', inplace=True, axis=1)
concatenated.drop(1,0)

concatenated.to_csv('ALL_E_train.csv')