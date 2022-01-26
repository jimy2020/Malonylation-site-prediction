import pandas as pd
import numpy as np

n_features_to_select = 100

df = pd.read_csv('../combined data/combined_weight_eaac_tfidf_dde_position_pka_M_train.csv')
# print(df.columns[0])
df.drop([df.columns[0]], inplace=True, axis=1)

T_data = []
F_data = []
i = 0
for key, row in df.iterrows():
    i += 1
    if i < (df.shape[0] / 2):
        T_data.append(row)
    else:
        F_data.append(row)

F_data = pd.DataFrame(data=F_data, columns=df.columns)
T_data = pd.DataFrame(data=T_data, columns=df.columns)

tot_mean = np.mean(df)
T_mean = np.mean(T_data)
F_mean = np.mean(F_data)

F_score = {}
for j in list(df.columns):
    SQE = ((T_mean[j] - tot_mean[j]) ** 2) + ((F_mean[j] - tot_mean[j]) ** 2)
    F_sum_q = 0
    T_sum_q = 0
    for ii in T_data.index:
        T_sum_q += (T_data.loc[ii, j] - T_mean[j]) ** 2
    for ii in F_data.index:
        F_sum_q += (F_data.loc[ii, j] - F_mean[j]) ** 2
    tot_sum = ((1 / (len(F_data) - 1)) * F_sum_q) + ((1 / (len(T_data) - 1)) * T_sum_q)
    F_score[j] = SQE / tot_sum

import matplotlib.pyplot as plt
plt.bar(F_score.keys(), F_score.values())
plt.savefig('bar plot.png')
plt.show()

# sorted_F_score = {k: v for k, v in sorted(F_score.items(), key=lambda item: item[1], reverse=True)}
# features_with_high_f_score = [i[0] for i in list(sorted_F_score.items())[:20]]
# print(features_with_high_f_score)

# dic = {k: v for k, v in enumerate(F_score) if k in features_with_high_f_score}
