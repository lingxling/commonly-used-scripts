# -*- coding: UTF-8 -*-

"""
训练随机森林模型
"""

import os
import utils
import copy
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


def main():
    if not os.path.exists('../model/'):
        os.mkdir('../model/')

    X_train, X_test, Y_train, Y_test = utils.load_train_and_test_data()
    label_class = ['normal', 'missing', 'minor', 'outlier', 'square', 'trend', 'drift']

    # 线性SVM
    print('--- 开始训练：随机森林分类器 ---')
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced')
    rf.fit(X_train, Y_train)
    pickle.dump(rf, open('../model/random_forest.model', 'wb'))

    pred = rf.predict(X_test)
    print(metrics.accuracy_score(Y_test, pred))
    return Y_test, pred


if __name__ == '__main__':
    main()