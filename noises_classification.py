from sklearn import model_selection, metrics
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('full_features.csv', sep=',')
data.drop(['Paths'], axis=1, inplace=True) # удаляем столбец с путями, т.к. они не нужны для прогноза


"""Разделяем данные на признаки и целевую переменную"""
y = data.Type_of_noise.values
data.iloc[:, :].drop(['Type_of_noise'], axis=1, inplace=True)
X = data


"""Кросс-валидация"""
cv = KFold(n_splits=7, shuffle=True, random_state=1)


def AdaBoost_classifier(X, y, cv):

    clf = AdaBoostClassifier(n_estimators=1000, random_state=17) # 0.70806371329
    clf.fit(X, y)
    print('AdaBoost: ', cross_val_score(clf, X, y, cv=cv).mean())
    print('----------------\n')


def decision_trees_classifier(X, y, cv):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)
    tree = DecisionTreeClassifier(random_state=17)   # объект модели
    tree.fit(X_train, y_train)
    tree_pred = tree.predict(X_test)
    print('Decision trees: ', accuracy_score(y_test, tree_pred)) # 0.625


    # Подкручиваем параметры: настройка максимальной глубины дерева
    tree_params = {'max_depth': range(1, 11)}
    tree_grid = GridSearchCV(tree, tree_params, cv=5, n_jobs=-1) # n_jobs=-1 <-- задействовать все ядра процессора
    tree_grid.fit(X_train, y_train)
    print('Decision trees, GridSearchCV-params: ', tree_grid.best_params_) # {'max_depth': 6}
    print('Decision trees, GridSearchCV-score: ', tree_grid.best_score_) # 0.660098522167
    print('Accuracy: ', accuracy_score(y_test, tree_grid.predict(X_test)))
    print('----------------\n')



def knn_classifier(X, y, cv):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    knn = KNeighborsClassifier()   # объект модели
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)
    print('kNN: ', accuracy_score(y_test, knn_pred)) # 0.534090909091


    # Попробуем настроить число соседей в алгоритме kNN
    knn_pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_jobs=-1))])
    knn_params = {'knn__n_neighbors':range(1, 10)}
    knn_grid = GridSearchCV(knn_pipe, knn_params, cv=5, n_jobs=-1)
    knn_grid.fit(X_train, y_train)
    print('kNN, GridSearchCV-params: ', knn_grid.best_params_) # {'knn__n_neighbors': 4}
    print('kNN, GridSearchCV-score: ', knn_grid.best_score_) # 0.738916256158
    print('Accuracy: ', accuracy_score(y_test, knn_grid.predict(X_test))) # 0.659090909091
    print('----------------\n')



def RFClassifier(X, y, cv):

    rf = RandomForestClassifier(n_estimators=100) # 0.780404844865
    rf.fit(X, y)
    print('RandomForestClassifier: ', cross_val_score(rf, X, y, cv=cv).mean())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    forest = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    forest_params = {'max_depth': range(1, 11)}
    forest_grid = GridSearchCV(forest, forest_params, cv=5, n_jobs=-1)
    forest_grid.fit(X_train, y_train)
    print('RandomForest, GridSearchCV-params: ', forest_grid.best_params_) #  {'max_depth': 9}
    print('RandomForest, GridSearchCV-score: ', forest_grid.best_score_) # 0.71921182266
    print('Accuracy: ', accuracy_score(y_test, forest_grid.predict(X_test))) # 0.784090909091
    print('----------------\n')




AdaBoost_classifier(X, y, cv)
RFClassifier(X, y, cv)
decision_trees_classifier(X, y, cv)
knn_classifier(X, y, cv)




