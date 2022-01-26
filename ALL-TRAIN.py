import pandas as pd
import numpy as np

Q=pd.read_csv(r'position_H_train.csv')
W=pd.read_csv(r'PKA_H_train.csv')
E=pd.read_csv(r'EAAC_H_train.csv')
R=pd.read_csv(r'weight_H_train.csv')
#T=pd.read_csv(r'KNN_H_train.csv')
#P=pd.read_csv(r'PKA_H_train.csv')
#Pr=pd.read_csv(r'pro2vec_1d_char_H_train.csv')
#Pro=pd.read_csv(r'pro2vec_embeding_char_H_train.csv')
#Ps=pd.read_csv(r'pssm_H_train.csv')
concatenated = pd.concat([E,W,Q,R], axis=1)
concatenated.drop('Unnamed: 0', inplace=True, axis=1)
concatenated.drop(1,0)

data1=np.matrix(concatenated[:])[:,:]
data_=pd.DataFrame(data=data1)
data_.to_csv('PPEW_combin_train.csv')