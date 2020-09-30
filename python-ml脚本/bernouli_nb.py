# -*- coding: UTF-8 -*-

"""
训练伯努利朴素贝叶斯模型
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

    # 朴素贝叶斯 - 伯努利
    print('--- 开始训练：伯努利朴素贝叶斯 ---')
    bernoulli_nb = naive_bayes.BernoulliNB()
    bernoulli_nb.fit(X_train, Y_train)
    pickle.dump(bernoulli_nb, open('../model/bernouli_nb.model', 'wb'))

    bernoulli_nb_pred = bernoulli_nb.predict(X_test)
    print(metrics.accuracy_score(Y_test, bernoulli_nb_pred))
    return Y_test, bernoulli_nb_pred


if __name__ == '__main__':
    main()
