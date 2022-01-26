import os
import sys

import numpy as np
import pandas as pd

from methods.DDE import DDE
from methods.MyWeight import Weight
from methods.EAAC import EAAC
from methods.PKA import PKA
from methods.TFIDF import TFIDF
from methods.position import Position
from methods.EGAAC import EGAAC
from combine import combine
from classifier.DNN import DNNClassifier
from classifier.DL import DLClassifier
from classifier.DL_1 import DL_1Classifier
from classifier.SVC_classifier import SVCClassifier
from classifier.GRU import GRUClassifier
from classifier.XGBoost_classifier import XGBoostClassifier
from itertools import combinations

dataFolder = r'data'
outputFolder = r'outputs'
TrainFileName = r'E_train'
TestFileName = r'E_test'


def weight():
    input_files = [dataFolder + r'/' + TrainFileName + '.txt', dataFolder + r'/' + TestFileName + '.txt']
    output_files = [dataFolder + r'/weight_' + TrainFileName + '.csv', dataFolder + r'/weight_' + TestFileName + '.csv']
    print('doing Weight ...')
    method = Weight()
    method.do(input_files, output_files)
    print('done')
    return [output_files[0], output_files[1]]


def eaac():
    input_files = [dataFolder + r'/' + TrainFileName + '.txt']
    output_files = [dataFolder + r'/EAAC_' + TrainFileName + '.csv']
    print('doing EAAC ...')
    method = EAAC()
    method.do(input_files, output_files)
    print('done')
    return output_files[0]


def tfidf():
    input_files = [dataFolder + r'/' + TrainFileName + '.txt']
    output_files = [dataFolder + r'/TFIDF_' + TrainFileName + '.csv']
    print('doing TFIDF ...')
    method = TFIDF()
    method.do(input_files, output_files)
    print('done')
    return output_files[0]


def dde():
    input_files = [dataFolder + r'/' + TrainFileName + '.txt']
    output_files = [dataFolder + r'/dde_' + TrainFileName + '.csv']
    print('doing dde ...')
    method = DDE()
    method.do(input_files, output_files)
    print('done')
    return output_files[0]


def pka():
    input_files = [dataFolder + r'/' + TrainFileName + '.txt']
    output_files = [dataFolder + r'/pka_' + TrainFileName + '.csv']
    print('doing pka ...')
    method = PKA()
    method.do(input_files, output_files)
    print('done')
    return output_files[0]


def position():
    input_files = [dataFolder + r'/' + TrainFileName + '.txt']
    output_files = [dataFolder + r'/position_' + TrainFileName + '.csv']
    print('doing position ...')
    method = Position()
    method.do(input_files, output_files)
    print('done')
    return output_files[0]


def egaac():
    input_files = [dataFolder + r'/' + TrainFileName + '.txt']
    output_files = [dataFolder + r'/egaac_' + TrainFileName + '.csv']
    print('doing egaac ...')
    method = EGAAC()
    method.do(input_files, output_files)
    print('done')
    return output_files[0]


def cor_selector(X, y, num_feats):
    cor_list = []
    feature_name = X.columns.tolist()

    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)

    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]

    # feature name
    cor_feature = X.iloc[:, np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()

    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature


def f_score(input_file, n_features_to_select):
    df = pd.read_csv(input_file)
    df.drop([df.columns[0]], inplace=True, axis=1)

    T_data = []
    F_data = []
    i = 0
    for key, row in df.iterrows():
        i += 1
        # if row[TARGET] == 1 or row[TARGET] == '1':
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

    sorted_F_score = {k: v for k, v in sorted(F_score.items(), key=lambda item: item[1], reverse=True)}
    features_with_high_f_score = [i[0] for i in list(sorted_F_score.items())[:n_features_to_select]]
    return features_with_high_f_score


def remove_duplicate(x):
    return list(dict.fromkeys(x))


def pp2(train_df, target_col, id_col):
    # corr = mean with target
    train_df.drop([id_col], axis=1, inplace=True, errors='ignore')
    features = train_df.columns.values[:]
    stds = train_df[features[1:]].std().abs()
    # corr_detail_print("std", stds)
    """
    corr all
    """
    correlations_all = train_df[features].corr().abs().unstack().sort_values(kind="quicksort").reset_index()
    correlations_all = correlations_all[correlations_all['level_0'] != correlations_all['level_1']]
    """
    corr with target
    """
    correlations_with_target = correlations_all[correlations_all['level_0'] == target_col]
    limit_for_corr = correlations_with_target[0].mean()
    # corr_detail_print("corr with target", correlations_with_target)
    features_keep_with_target = [f for f in
                                 (correlations_with_target[correlations_with_target[0] >= limit_for_corr])['level_1']]
    features_keep_with_target = remove_duplicate(features_keep_with_target)
    features_keep = list(set(features_keep_with_target + [target_col]))
    print('features_keep:', len(features_keep))
    features_remove = [f for f in features if f not in features_keep]
    print('features_remove:', len(features_remove))
    if len(features_remove) > 0:
        train_df.drop(features_remove, axis=1, inplace=True, errors='ignore')
    return train_df


def dnn_test(data_):
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import scale
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import StratifiedKFold
    import utils.tools as utils
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Flatten
    import numpy as np
    from keras.layers import Conv1D, MaxPooling1D, Dropout
    from keras.models import Model

    model = Sequential()
    model.add(Dense(32, activation='relu', name="Dense_32"))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu', name="Dense_16"))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu', name="Dense_8"))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu', name="Dense_4"))
    model.add(Dense(2, activation='softmax', name="Dense_2"))
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])  # rmsprop

    data = np.array(data_)
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

    skf = StratifiedKFold(n_splits=10)

    for train, test in skf.split(X, y):
        y_train = utils.to_categorical(y[train])  # generate the resonable results
        cv_clf = model
        hist = cv_clf.fit(X[train],
                          y_train,
                          verbose=0,
                          epochs=19)

        y_score = cv_clf.predict(X[test])  # the output of  probability
        y_class = utils.categorical_probas_to_classes(y_score)

        y_test = utils.to_categorical(y[test])  # generate the test
        ytest = np.vstack((ytest, y_test))
        y_test_tmp = y[test]
        yscore = np.vstack((yscore, y_score))

        acc, precision, npv, sensitivity, specificity, mcc, f1 = utils.calculate_performace(len(y_class), y_class,
                                                                                            y_test_tmp)
        # fpr, tpr, _ = roc_curve(y_test[:,0], y_score[:,0])
        fpr, tpr, _ = roc_curve(y_test[:, 0], y_score[:, 0])
        roc_auc = auc(fpr, tpr)
        y_class = utils.categorical_probas_to_classes(y_score)
        acc, precision, npv, sensitivity, specificity, mcc, f1 = utils.calculate_performace(len(y_class), y_class,
                                                                                            y_test_tmp)
        sepscores.append([acc, precision, npv, sensitivity, specificity, mcc, f1, roc_auc])
        print('DNN:acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,f1=%f,roc_auc=%f'
              % (acc, precision, npv, sensitivity, specificity, mcc, f1, roc_auc))

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
    yscore = yscore[np.array(range(1, row)), :]
    yscore_sum = pd.DataFrame(data=yscore)
    # yscore_sum.to_csv('outputs/' + classifier_name + str(
    #     (np.mean(scores, axis=0)[0] * 100, np.std(scores, axis=0)[0] * 100)) + '_yscore_' + input_file_name)
    ytest = ytest[np.array(range(1, row)), :]
    ytest_sum = pd.DataFrame(data=ytest)
    # ytest_sum.to_csv('outputs/' + classifier_name + str(
    #     (np.mean(scores, axis=0)[0] * 100, np.std(scores, axis=0)[0] * 100)) + '_ytest_' + input_file_name)
    fpr, tpr, _ = roc_curve(ytest[:, 0], yscore[:, 0])
    auc_score = np.mean(scores, axis=0)[7]
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='DNN ROC (area = %0.2f%%)' % auc_score)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    data_csv = pd.DataFrame(data=result)
    # data_csv.to_csv('outputs/' + classifier_name + str(
    #     (np.mean(scores, axis=0)[0] * 100, np.std(scores, axis=0)[0] * 100)) + '_data_' + input_file_name)


def main2():
    w = weight()
    e = eaac()
    t = tfidf()
    d = dde()
    pos = position()
    pk = pka()

    # df = pd.read_csv(w[0])
    # df.drop(['Unnamed: 0'], axis=1, inplace=True)  # removes in df itself
    # X = df.drop(['0'], axis=1)
    # y = df['0']
    # cor_support, cor_feature = cor_selector(X, y, 10)

    combine_output = r'combined data/combined_' + TrainFileName + '.csv'
    combine([w, e, t,d,pos,pk], output_file_name=combine_output)
    #DNNClassifier(combine_output)
    XGBoostClassifier(combine_output)
    # DNN(pk)
    pass


def main3():
    
    all_data = [
        ['egaac', egaac()],
        ['weight', weight()],
        ['eaac', eaac()],
        ['tfidf', tfidf()],
        ['dde', dde()],
        ['position', position()],
        ['pka', pka()],
    ]

    classifiers = [
        DNNClassifier,
        # DLClassifier,
        # DL_1Classifier,
        GRUClassifier,
        XGBoostClassifier,
        SVCClassifier,
    ]

    # combines_outputs = []
    idx = 2
    all_combinations = [list(combinations(all_data, i)) for i in range(1, len(all_data) + 1)]
    total_combinations = sum([len(all_combinations[i]) for i in range(len(all_combinations))])
    for combination in all_combinations:
        for methods in combination:
            print(idx, 'of', total_combinations)
            idx += 1
            methods_name = ''
            for method in methods:
                methods_name += '_' + method[0]
            combine_output = r'combined data/combined' + methods_name + '_' + TrainFileName + '.csv'
            if not os.path.exists(combine_output):
                print('combining', methods_name)
                combine([method[1] for method in methods], output_file_name=combine_output)
                print('done')
            else:
                print('was combined', combine_output)
            # combines_outputs.append(combine_output)

            # data = pd.read_csv(combine_output)
            # data = np.array(data)
            # data = data[:, 1:]
            # [m1, n1] = np.shape(data)
            # label1 = np.ones((int(m1 / 2), 1))  # Value can be changed
            # label2 = np.zeros((int(m1 / 2), 1))
            # label = np.append(label1, label2)

            # df = pp2(pd.read_csv(combine_output), label, 'Unnamed: 0')
            # df.to_csv('test.csv')

            for clf in classifiers:
                print('starting', clf.__name__)
                clf(combine_output)
                # clf('test.csv')
                print('done')
                sys.exit(-1)


def main4():
    from sklearn.manifold import TSNE
    df = pd.read_csv('./combined data/combined_tfidf_pka_M_train.csv')
    df.drop('Unnamed: 0', axis=1, inplace=True)
    dnn_test(df)
    tsne = TSNE(n_components=3, n_iter=300).fit_transform(df.values)
    dnn_test(tsne)


if __name__ == '__main__':
    #f_score()
     main3()
    #input_file = 'C:/Users/parsian/Downloads/WETPE2_min-max_train.csv';
    #input_file = 'E:/DTI_Girls/Doc_Files/DTI-gitlab/data/EsixfeatureW-min-max_combin_train.csv'
    # DNNClassifier(input_file)
    #features_selected = f_score(input_file,120)
    #DNNClassifier(input_file, features=features_selected)
