# -*- coding: UTF-8 -*-

"""
展示结果
"""

from prettytable import PrettyTable
from collections import Counter
import copy


def main(res):
    y_true = res[0] + 1
    y_pred = res[1] + 1
    classes = ['normal', 'missing', 'minor', 'outlier', 'square', 'trend', 'drift', 'accuracy']
    table = PrettyTable(['class', 'normal', 'missing', 'minor', 'outlier', 'square', 'trend', 'drift', 'accuracy'])
    for i in range(1, 8):
        # [1, 2, 3, 4, 5, 6, 7]
        cur_true = copy.deepcopy(y_true)
        cur_pred = copy.deepcopy(y_pred)

        # 真实类的当前类设为-1，其余为0
        cur_true[cur_true == i] = -1
        cur_true[cur_true != -1] = 0
        total_true_number = abs(sum(cur_true))

        cur_pred[cur_true == 0] = 0  # 真实类的非当前类设为0
        # 计算各类的分布情况
        cnt = Counter(cur_pred)
        cur_class_name = classes[i - 1]
        row = [cur_class_name]
        for k in range(1, 8):
            if k not in cnt.keys():
                row.append('0 0.0%')
                continue
            perc = cnt[k] / total_true_number * 100
            row.append(str(cnt[k]) + ' %.2f%%' % perc)

        perc = cnt[i] / total_true_number * 100
        row.append('%.2f%%' % perc)  # 准确率
        table.add_row(row)
    print(table)
    pass
