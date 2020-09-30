# -*- coding: UTF-8 -*-

"""
训练线性SVM模型
"""

import os
import utils
import copy
import pickle
import numpy as np
from sklearn import svm
from sklearn import metrics


def main():
    if not os.path.exists('../model/'):
        os.mkdir('../model/')

    X_train, X_test, Y_train, Y_test = utils.load_train_and_test_data()
    label_class = ['normal', 'missing', 'minor', 'outlier', 'square', 'trend', 'drift']

    # 线性SVM
    print('--- 开始训练：线性SVM分类器 ---')
    linear_svc = svm.LinearSVC(max_iter=5000)
    linear_svc.fit(X_train, Y_train)
    pickle.dump(linear_svc, open('../model/linear_svc.model', 'wb'))

    linear_svc_pred = linear_svc.predict(X_test)
    print(metrics.accuracy_score(Y_test, linear_svc_pred))
    return Y_test, linear_svc_pred


if __name__ == '__main__':
    main()

