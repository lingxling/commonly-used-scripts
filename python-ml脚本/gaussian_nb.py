# -*- coding: UTF-8 -*-

"""
训练高斯朴素贝叶斯模型
"""

import os
import utils
import copy
import pickle
import numpy as np
from sklearn import svm, naive_bayes
from sklearn import metrics


def main():
    if not os.path.exists('../model/'):
        os.mkdir('../model/')

    X_train, X_test, Y_train, Y_test = utils.load_train_and_test_data()
    label_class = ['normal', 'missing', 'minor', 'outlier', 'square', 'trend', 'drift']

    # 朴素贝叶斯 - 高斯分布
    print('--- 高斯朴素贝叶斯 ---')
    gaussian_nb = naive_bayes.GaussianNB()
    gaussian_nb.fit(X_train, Y_train)
    pickle.dump(gaussian_nb, open('../model/gaussian_nb.model', 'wb'))

    gaussian_nb_pred = gaussian_nb.predict(X_test)
    print(metrics.accuracy_score(Y_test, gaussian_nb_pred))
    return Y_test, gaussian_nb_pred


if __name__ == '__main__':
    main()
