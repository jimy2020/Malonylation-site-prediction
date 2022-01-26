import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
import pandas as pd


def GBDT_tuning(x, y):
    parameters = {'n_estimators': [200, 300, 400, 500], 'learning_rate': [0.1, 0.05, 0.2], 'max_depth': [10, 15, 20]}

    print("# Tuning hyper-parameters ")
    print()

    clf = GridSearchCV(GradientBoostingClassifier(), parameters, cv=5,
                       scoring='roc_auc')
    clf.fit(x, y)
    print("Best parameters set found on development set:")
    return clf.best_params_, clf.grid_scores_


def RF_tuning(x, y):
    parameters = {'n_estimators': [200, 300, 400, 500], 'max_depth': [10, 15, 20]}

    print("# Tuning hyper-parameters ")
    print()

    clf = GridSearchCV(RandomForestClassifier(), parameters, cv=5,
                       scoring='roc_auc')

    clf.fit(x, y)
    print("Best parameters set found on development set:")
    return clf.best_params_, clf.grid_scores_


def cross_validation(x, y):
    names = ["GBDT", "RF", "DNN"]
    classifiers = [GradientBoostingClassifier(learning_rate=0.1, n_estimators=400, max_depth=10),
                   RandomForestClassifier(n_estimators=500, max_depth=25),
                   DNN()]
    model_metrics_name = [accuracy_score, precision_score, recall_score, f1_score]

    pre_y_list = []
    fpr_list = []
    tpr_list = []
    auc_list = []

    for name, clf in zip(names, classifiers):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=99)
        clf = clf.fit(x_train, y_train)
        pre_y_list.append(clf.predict(x_test))
        pre_y_proba = clf.predict_proba(x_test)
        fpr_, tpr_, _ = roc_curve(y_test, pre_y_proba, pos_label=1)
        auc_ = auc(fpr_, tpr_)
        fpr_list.append(fpr_)
        tpr_list.append(tpr_)
        auc_list.append(auc_)

    n_samples, n_features = x.shape 
    model_metrics_list = [] 
    for i in range(5):
        tmp_list = [] 
        for m in model_metrics_name: 
            tmp_score = m(y_test, pre_y_list[i]) 
            tmp_list.append(tmp_score)  
            model_metrics_list.append(tmp_list)  
            # cv_score_df=pd.DataFrame(cv_score_list,index=names)   
            model_metrics_df = pd.DataFrame(model_metrics_list, index=names,
                                            columns=['acc', 'pre', 'rec', 'f1'])  
    return fpr_list, tpr_list, auc_list, model_metrics_df
