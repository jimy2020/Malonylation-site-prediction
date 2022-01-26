import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import utils.tools as utils
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv1D, MaxPooling1D, Dropout
from keras.layers import LSTM
import warnings
warnings.filterwarnings("ignore")
NumFeRedu = 500
NumModel = 8
n_pca = 13
if NumModel == 10:
    NumFeRedu = 100

def f_score(input_file, n_features_to_select):
    df = pd.read_csv(input_file)
    df.drop([df.columns[0]], inplace=True, axis=1)

    T_data = []
    F_data = []
    i = 0
    for key, row in df.iterrows():
        i += 1
        # if row[TARGET] == 1 or row[TARGET] == '1':
        if i <= (df.shape[0] / 2):
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

    sorted_F_score = {k: v for k, v in sorted(F_score.items(), key=lambda item: item[1], reverse=True)}
    features_with_high_f_score = [i[0] for i in list(sorted_F_score.items())[:n_features_to_select]]
    return features_with_high_f_score
 
    
def featureReduction(NumFeRedu, X_train, X_test, X_inde, y_train):
    print('NumFeRedu', NumFeRedu)
    if NumFeRedu == 1 or NumFeRedu == 3:
        print('dffdfdfdfdfffdfdfd')
        pca = PCA(n_components=n_pca)
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
        X_inde = pca.transform(X_inde)
    if NumFeRedu == 2 or NumFeRedu == 3:
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, y_train)
        X_train = lda.transform(X_train)
        X_test = lda.transform(X_test)
        X_inde = lda.transform(X_inde)
    if NumFeRedu == 4:
        FICA = FastICA(n_components=1000, random_state=0)
        FICA.fit(X_train)
        X_train = FICA.transform(X_train)
        X_test = FICA.transform(X_test)
        X_inde = FICA.transform(X_inde)
    return X_train, X_test, X_inde


def createMode(NumModel):
    if NumModel == 0:
        model = xgb.XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=6)
    if NumModel == 1:
        model = LogisticRegression(solver='liblinear')
    if NumModel == 2:
        model = SVC(kernel='rbf', C=1, gamma='auto')
    if NumModel == 3:
        model = DecisionTreeClassifier()
    if NumModel == 4:
        model = RandomForestClassifier(n_estimators=1000)
    if NumModel == 5:
        model = AdaBoostClassifier(n_estimators=100)
    if NumModel == 6:
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0)
    if NumModel == 7:
        model = GaussianNB()
    if NumModel == 8:
        clf1 = LogisticRegression(solver='lbfgs', multi_class='ovr', random_state=1)
        clf2 = RandomForestClassifier(n_estimators=100, random_state=1)
        clf3 = GaussianNB()
        clf4 = SVC(kernel='linear', probability=True)
        model = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3), ('svc', clf4)], voting='soft',
                                 weights=[1, 1, 1, 1])
    if NumModel == 9:
        model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    if NumModel == 10:
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=4))
        model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=4))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', name="Dense_128"))
        model.add(Dense(64, activation='relu', name="Dense_64"))
        model.add(Dense(32, activation='relu', name="Dense_32"))
        model.add(Dense(2, activation='softmax', name="Dense_2"))
        model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])  # rmsprop
    if NumModel == 11:
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

input_file =r'D:/Paper/DTI_Girls/Doc_Files/DTI-gitlab/combined data/combined_E_train.csv'
data_1 = pd.read_csv(input_file)
data = np.array(data_1)
data = data[:, 1:]
[m1, n1] = np.shape(data)
label1 = np.ones((int(m1 / 2), 1))  # Value can be changed
label2 = np.zeros((int(m1 / 2), 1))
label = np.append(label1, label2)
shu = scale(data)
X = shu
y = label

sepscores = []

ytest = np.ones((1, 2)) * 0.5
yscore = np.ones((1, 2)) * 0.5
# 53 27
if NumModel == 10:
    X = X.reshape(-1, 1, 470)
if NumModel == 11:
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

print(X.shape)
#######################################################
data_1 = pd.read_csv(r'D:/Paper/DTI_Girls/Doc_Files/DTI-gitlab/combined data/combined_E_train.csv')
data1 = np.array(data_1)
data1 = data1[:, 1:]
[m11, n11] = np.shape(data1)
label11 = np.ones((int(m11 / 2), 1))  # Value can be changed
label21 = np.zeros((int(m11 / 2), 1))
label1 = np.append(label11, label21)
shu1 = scale(data1)
X1 = shu1
y1 = label1

sepscores1 = []

ytest1 = np.ones((1, 2)) * 0.5
yscore1 = np.ones((1, 2)) * 0.5
# 53 27
if NumModel == 10:
    X1 = X1.reshape(-1, 1, 470)
if NumModel == 11:
    X1 = np.reshape(X1, (X1.shape[0], X1.shape[1], 1))

epc = 19
skf = StratifiedKFold(n_splits=10, shuffle=True)

for train, test in skf.split(X, y):
    print('train  ', X[train].shape)
    print('test   ', X[test].shape)
    print('ind   ', X1.shape)
    '''scaler = MinMaxScaler().fit(X[train])
    X[train]=scaler.transform(X[train])
    X[test]=scaler.transform(X[test])
    X1=scaler.transform(X1)'''
    y_train = utils.to_categorical(y[train])  # generate the resonable results
    X_train, X_test, X_inde = featureReduction(NumFeRedu, X[train], X[test], X1, y[train])
    Renk_feature=f_score(input_file, 20);
    modelNew = createMode(NumModel)
    if NumModel == 10:
        modelNew.fit(X_train, y_train, epochs=epc)
    elif NumModel == 11:
        modelNew.fit(X_train, y[train], epochs=epc, batch_size=32)
    else:
        modelNew.fit(X_train, y[train])
    y_score = modelNew.predict(X_test)
    ytest = utils.to_categorical(y[test])
    y_score = utils.to_categorical(y_score)
    y_test_tmp = y[test]
    fpr, tpr, _ = roc_curve(ytest[:, 0], y_score[:, 0])
    roc_auc = auc(fpr, tpr)
    y_class = utils.categorical_probas_to_classes(y_score)
    acc, precision, npv, sensitivity, specificity, mcc, f1 = utils.calculate_performace(len(y_class), y_class,
                                                                                        y_test_tmp)
    sepscores.append([acc, precision, npv, sensitivity, specificity, mcc, f1, roc_auc])
    print('GTB:acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,f1=%f,roc_auc=%f'
          % (acc, precision, npv, sensitivity, specificity, mcc, f1, roc_auc))

    print('#############################################################')
    y_test1 = utils.to_categorical(y1)  # generate the test
    y_test_tmp1 = y1
    print(X_inde.shape)
    y_score1 = modelNew.predict(X_inde)  # the output of  probability
    y_score1 = utils.to_categorical(y_score1)
    fpr1, tpr1, _ = roc_curve(y_test1[:, 0], y_score1[:, 0])
    roc_auc1 = auc(fpr1, tpr1)
    y_class1 = utils.categorical_probas_to_classes(y_score1)
    acc1, precision1, npv1, sensitivity1, specificity1, mcc1, f11 = utils.calculate_performace(len(y_class1), y_class1,
                                                                                               y_test_tmp1)
    sepscores1.append([acc1, precision1, npv1, sensitivity1, specificity1, mcc1, f11, roc_auc1])
    print('GTB:acc1=%f,precision1=%f,npv1=%f,sensitivity1=%f,specificity1=%f,mcc=%f,f11=%f,roc_auc1=%f'
          % (acc1, precision1, npv1, sensitivity1, specificity1, mcc1, f11, roc_auc1))
    print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')

print('Total Result  %%%%%%')
scores = np.array(sepscores)
print("acc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[0] * 100, np.std(scores, axis=0)[0] * 100))
print("precision=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[1] * 100, np.std(scores, axis=0)[1] * 100))
print("npv=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[2] * 100, np.std(scores, axis=0)[2] * 100))
print("sensitivity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[3] * 100, np.std(scores, axis=0)[3] * 100))
print("specificity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[4] * 100, np.std(scores, axis=0)[4] * 100))
print("mcc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[5] * 100, np.std(scores, axis=0)[5] * 100))
print("f1=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[6] * 100, np.std(scores, axis=0)[6] * 100))
print("roc_auc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[7] * 100, np.std(scores, axis=0)[7] * 100))
result1 = np.mean(scores, axis=0)
H1 = result1.tolist()
sepscores.append(H1)
result = sepscores
row = yscore.shape[0]
'''
yscore=yscore[np.array(range(1,row)),:]
yscore_sum = pd.DataFrame(data=yscore)
yscore_sum.to_csv('yscore_DL_E_train_19.csv')
ytest=ytest[np.array(range(1,row)),:]
ytest_sum = pd.DataFrame(data=ytest)
ytest_sum.to_csv('ytest_DL_E_train_19.csv')
fpr, tpr, _ = roc_curve(ytest[:,0], yscore[:,0])
auc_score=np.mean(scores, axis=0)[7]
lw=2
plt.plot(fpr, tpr, color='darkorange',
lw=lw, label='DL ROC (area = %0.2f%%)' % auc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
data_csv = pd.DataFrame(data=result)
data_csv.to_csv('cnn_dnn_E_train_19.csv')'''
''''
dense1_layer_model = Model(inputs=model.input,outputs=model.get_layer('Dense_2').output)
dense1_output = dense1_layer_model.predict(X)
data_csv1 = pd.DataFrame(data=dense1_output)
data_csv1.to_csv('dense2_E_train_19.csv')


dense2_layer_model = Model(inputs=model.input,outputs=model.get_layer('Dense_32').output)
dense2_output = dense2_layer_model.predict(X)
data_csv2 = pd.DataFrame(data=dense2_output)
data_csv2.to_csv('dense32_E_train_19.csv')

dense3_layer_model = Model(inputs=model.input,outputs=model.get_layer('Dense_64').output)
dense3_output = dense3_layer_model.predict(X)
data_csv3 = pd.DataFrame(data=dense3_output)
data_csv3.to_csv('dense64_E_train_19.csv')

dense4_layer_model = Model(inputs=model.input,outputs=model.get_layer('Dense_128').output)
dense4_output = dense4_layer_model.predict(X)
data_csv4 = pd.DataFrame(data=dense4_output)
data_csv4.to_csv('dense128_E_train_19.csv')
'''

####################################################
scores1 = np.array(sepscores1)
print("acc=%.2f%% (+/- %.2f%%)" % (np.mean(scores1, axis=0)[0] * 100, np.std(scores1, axis=0)[0] * 100))
print("precision=%.2f%% (+/- %.2f%%)" % (np.mean(scores1, axis=0)[1] * 100, np.std(scores1, axis=0)[1] * 100))
print("npv=%.2f%% (+/- %.2f%%)" % (np.mean(scores1, axis=0)[2] * 100, np.std(scores1, axis=0)[2] * 100))
print("sensitivity=%.2f%% (+/- %.2f%%)" % (np.mean(scores1, axis=0)[3] * 100, np.std(scores1, axis=0)[3] * 100))
print("specificity=%.2f%% (+/- %.2f%%)" % (np.mean(scores1, axis=0)[4] * 100, np.std(scores1, axis=0)[4] * 100))
print("mcc=%.2f%% (+/- %.2f%%)" % (np.mean(scores1, axis=0)[5] * 100, np.std(scores1, axis=0)[5] * 100))
print("f1=%.2f%% (+/- %.2f%%)" % (np.mean(scores1, axis=0)[6] * 100, np.std(scores1, axis=0)[6] * 100))
print("roc_auc=%.2f%% (+/- %.2f%%)" % (np.mean(scores1, axis=0)[7] * 100, np.std(scores1, axis=0)[7] * 100))
result11 = np.mean(scores1, axis=0)
H11 = result11.tolist()
sepscores1.append(H11)
result1 = sepscores1
row1 = yscore1.shape[0]
'''
yscore1=yscore[np.array(range(1,row1)),:]
yscore_sum1 = pd.DataFrame(data=yscore1)
yscore_sum1.to_csv('yscore1_DL_E_train_19.csv')
ytest1=ytest1[np.array(range(1,row1)),:]
ytest_sum1 = pd.DataFrame(data=ytest1)
ytest_sum1.to_csv('ytest1_DL_E_train_19.csv')
fpr1, tpr1, _ = roc_curve(ytest1[:,0], yscore1[:,0])
auc_score1=np.mean(scores1, axis=0)[7]
lw1=2
plt.plot(fpr1, tpr1, color='darkorange',
lw=lw1, label='DL ROC (area = %0.2f%%)' % auc_score1)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
data_csv = pd.DataFrame(data=result1)
data_csv.to_csv('cnn_dnn_E_train_19.csv')
'''
'''
dense1_layer_model1 = Model(inputs=model.input,outputs=model.get_layer('Dense_2').output)
dense1_output1 = dense1_layer_model1.predict(X)
data_csv11 = pd.DataFrame(data=dense1_output1)
data_csv11.to_csv('dense21_E_train_19.csv')


dense2_layer_model1 = Model(inputs=model.input,outputs=model.get_layer('Dense_32').output)
dense2_output1 = dense2_layer_model1.predict(X)
data_csv21 = pd.DataFrame(data=dense2_output1)
data_csv21.to_csv('dense321_E_train_19.csv')

dense3_layer_model1 = Model(inputs=model.input,outputs=model.get_layer('Dense_64').output)
dense3_output1 = dense3_layer_model1.predict(X)
data_csv31 = pd.DataFrame(data=dense3_output1)
data_csv31.to_csv('dense641_E_train_19.csv')

dense4_layer_model = Model(inputs=model.input,outputs=model.get_layer('Dense_128').output)
dense4_output = dense4_layer_model.predict(X)
data_csv4 = pd.DataFrame(data=dense4_output)
data_csv4.to_csv('dense128_E_train_19.csv')
'''
