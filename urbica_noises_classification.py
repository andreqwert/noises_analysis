from sklearn import model_selection, metrics
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import KFold, cross_val_score



data = pd.read_csv('full_features.csv', sep=',')
data.drop(['Paths'], axis=1, inplace=True) # удаляем столбец с путями, т.к. они не нужны для прогноза


"""Разделяем данные на признаки и целевую переменную"""
y = data.Type_of_noise.values
data.iloc[:, :].drop(['Type_of_noise'], axis=1, inplace=True)
X = data


"""Кросс-валидация"""
cv = KFold(n_splits=7, shuffle=True, random_state=1)



def AdaBoost_classifier(X, y, cv):

    clf = AdaBoostClassifier(n_estimators=1000) # 0.70806371329
    clf.fit(X, y)
    print('AdaBoost: ', cross_val_score(clf, X, y, cv=cv).mean())



def RFClassifier(X, y, cv):

    rf = RandomForestClassifier(n_estimators=100) # 0.780404844865
    rf.fit(X, y)
    print('RandomForestClassifier: ', cross_val_score(rf, X, y, cv=cv).mean())



AdaBoost_classifier(X, y, cv)
RFClassifier(X, y, cv)





