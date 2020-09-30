# -*- coding: utf-8 -*-

import os
import h5py
import copy
import scipy.io as sio
import numpy as np
from scipy import fft
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter


def get_dir_path_list():
    base_path = "../data/"
    dir_path_list = []

    fft_path = '../FFT_data/'
    if not os.path.exists(fft_path):
        os.mkdir(fft_path)
    fft_dir_path_list = []

    # 获得所有文件夹的路径；创建对应的FFT文件夹
    for direc in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, direc)):
            dir_path_list.append(os.path.join(base_path, direc))
            fft_dir_path = os.path.join(fft_path, direc)
            if not os.path.exists(fft_dir_path):
                os.mkdir(fft_dir_path)
            fft_dir_path_list.append(fft_dir_path)
    return dir_path_list, fft_dir_path_list


def simple_preprocessed(begin=0, end=30):
    """
    Warning: 全部数据保存需要大概20G硬盘存储空间. 建议每次只保存1天的数据，用完后调用delete删掉。
    1. 用平均值填补缺失值，去掉全nan样例。
    2. 处理label，去掉那些全nan样例对应的label，并把label添加到sample的最后
    """
    print("simple preprocessed month", begin, '-', end)
    dir_path_list, fft_dir_path_list = get_dir_path_list()
    labels_filepath = '../data/label.mat'
    labelfile = h5py.File(labels_filepath)
    hour_cnt = 0  # [0, 744)

    for dir_path, fft_dir_path in zip(dir_path_list[begin:end], fft_dir_path_list[begin:end]):
        for filename in os.listdir(dir_path):
            if filename[-4:] == '.mat':  # 处理.mat文件
                filepath = os.path.join(dir_path, filename)
                data = sio.loadmat(filepath)['data'].T  # shape = (38, 72000)
                new_data = []
                sensor = -1  # [0, 38)

                for sample in data:
                    sensor += 1
                    ref = labelfile['info']['label']['manual'][sensor][0]
                    label = labelfile[ref][hour_cnt][0] - 1  # 0, 1, 2, 3, 4, 5, 6

                    if np.isnan(sample).sum() >= 40000:
                        sample[np.isnan(sample)] = 0  # 对于缺失值过多的样例(missing类)，补0
                    else:
                        ## median = np.nanmedian(sample)
                        ## sample[np.isnan(sample)] = median  # 用中位数填补缺失值
                        mean = np.nanmean(sample)
                        sample[np.isnan(sample)] = mean  # 用平均值填补缺失值

                    new_sample = np.append(copy.deepcopy(sample), label)
                    new_data.append(new_sample)

                new_data = np.array(new_data)
                filepath = os.path.join(dir_path, filename[:-4])  # 去掉.mat后缀
                np.save(filepath, new_data)

                hour_cnt += 1


def sample_preprocessed(step=72):
    """
    1. 用平均值填补缺失值，去掉全nan样例。
    2. 每20个采样只保留振幅最大的，把维度降到3600
    3. 处理label，去掉那些全nan样例对应的label，并把label添加到sample的最后
    """
    print("sample preprocessed")

    dir_path_list, fft_dir_path_list = get_dir_path_list()
    labels_filepath = '../data/label.mat'
    labelfile = h5py.File(labels_filepath)
    hour_cnt = 0  # [0, 744)

    for dir_path, fft_dir_path in zip(dir_path_list, fft_dir_path_list):
        for filename in os.listdir(dir_path):
            if filename[-4:] == '.mat':  # 处理.mat文件
                filepath = os.path.join(dir_path, filename)
                data = sio.loadmat(filepath)['data'].T  # shape = (38, 72000)
                new_data = []
                sensor = -1  # [0, 38)

                for sample in data:
                    sensor += 1
                    ref = labelfile['info']['label']['manual'][sensor][0]
                    label = labelfile[ref][hour_cnt][0] - 1
                    if np.isnan(sample).sum() >= 40000:
                        sample[np.isnan(sample)] = 0  # 对于缺失值过多的样例(missing类)，补0
                    else:
                        ## median = np.nanmedian(sample)
                        ## sample[np.isnan(sample)] = median  # 用中位数填补缺失值
                        mean = np.nanmean(sample)
                        sample[np.isnan(sample)] = mean  # 用平均值填补缺失值
                    fft_sample = fft(sample)

                    new_sample = []
                    new_fft_sample = []
                    for start in range(0, len(sample), step):  # 每隔step个数据取一个值
                        new_sample.append(np.mean(sample[start:start+step]))
                        new_fft_sample.append(np.mean(fft_sample[start:start+step]))
                        # new_sample.append(sample[start])
                        # new_fft_sample.append(fft_sample[start])
                        ## new_sample.append(max(sample[start:start+step]))
                        ## new_fft_sample.append(max(fft_sample[start:start+step]))
                    # new_fft_sample = fft(new_sample)
                    new_sample.extend(abs(np.array(new_fft_sample)))
                    new_sample.append(label)
                    new_data.append(new_sample)

                new_data = np.array(new_data)
                filepath = os.path.join(dir_path, filename[:-4])  # 去掉.mat后缀
                np.save(filepath, new_data)
                hour_cnt += 1


def load_preprocessed_data(begin=0, end=30):
    """
    载入预处理后的数据，注意避免内存溢出错误
    :return: sample, label
    """
    print('load preprocessed data')
    data = []
    dir_path_list, fft_dir_path_list = get_dir_path_list()
    for dir_path, fft_dir_path in zip(dir_path_list[begin:end], fft_dir_path_list[begin:end]):
        for filename in os.listdir(dir_path):
            if filename[-4:] == '.npy':  # 处理.npy文件
                filepath = os.path.join(dir_path, filename)
                data.append(np.load(filepath))
    data = np.array(data)
    shape = data.shape  # (744, 38, 72000)
    data = np.reshape(data, (shape[0]*shape[1], shape[2]))
    samples = data[:, :-1]
    labels = data[:, -1].astype(int)
    return samples, labels


def get_slice_data(new_shape=(120, 600), begin=0, end=30):
    """
    将数据切片
    :param new_shape: (切片数量，切片后数据的维度)
    :param begin: begin day
    :param end: end day
    :return: samples, labels (每个样例的第一个维度是时序，第二个是频谱)
    """
    print('get slice data')
    samples, labels = load_preprocessed_data(begin, end)
    new_samples = []
    new_labels = []
    for sample, label in zip(samples, labels):
        slice_sample_list = np.reshape(sample, newshape=new_shape)
        for slice_sample in slice_sample_list:
            fft_result = fft(slice_sample)
            fft_result[0] = 0
            new_sample = np.append(slice_sample, abs(fft_result))
            new_samples.append(new_sample)
        new_labels.extend([label]*new_shape[0])
    new_labels = np.array(new_labels)
    new_samples = np.array(new_samples)

    # 将Y编码为one-hot
    encoder = LabelEncoder()
    new_labels = encoder.fit_transform(new_labels)
    new_labels = np_utils.to_categorical(new_labels)

    return new_samples, new_labels


def delete_data(begin=0, end=30):
    """
    删除data和fft_data中的.npy数据，节约硬盘空间
    """
    print('delete data in fft_data and data')
    dir_path_list, fft_dir_path_list = get_dir_path_list()
    for dir_path, fft_dir_path in zip(dir_path_list[begin:end], fft_dir_path_list[begin:end]):
        for filename in os.listdir(dir_path):
            if filename[-4:] == '.npy':  # 删除.npy文件
                filepath = os.path.join(dir_path, filename)
                fft_filepath = os.path.join(fft_dir_path, filename)
                if os.path.exists(filepath):
                    os.remove(filepath)
                if os.path.exists(fft_filepath):
                    os.remove(fft_filepath)


def save_train_and_test_data(samples, labels):
    print('save train and test data')
    train_and_data_path = '../train_and_test_data/'
    if not os.path.exists(train_and_data_path):
        os.mkdir(train_and_data_path)
    X_train_file = os.path.join(train_and_data_path, 'X_train')
    X_test_file = os.path.join(train_and_data_path, 'X_test')
    Y_train_file = os.path.join(train_and_data_path, 'Y_train')
    Y_test_file = os.path.join(train_and_data_path, 'Y_test')

    X_train, X_test, Y_train, Y_test = train_test_split(samples, labels, shuffle=True, test_size=0.2)
    np.save(X_train_file, X_train)
    np.save(X_test_file, X_test)
    np.save(Y_train_file, Y_train)
    np.save(Y_test_file, Y_test)


def load_train_and_test_data(train_and_data_path='../train_and_test_data/'):
    print('load train and test data')
    X_train_file = os.path.join(train_and_data_path, 'X_train.npy')
    X_test_file = os.path.join(train_and_data_path, 'X_test.npy')
    Y_train_file = os.path.join(train_and_data_path, 'Y_train.npy')
    Y_test_file = os.path.join(train_and_data_path, 'Y_test.npy')

    X_train = np.load(X_train_file)
    X_test = np.load(X_test_file)
    Y_train = np.load(Y_train_file)
    Y_test = np.load(Y_test_file)
    return X_train, X_test, Y_train, Y_test


def main():
    labels_filepath = '../data/label.mat'
    labelfile = h5py.File(labels_filepath)
    pass


if __name__ == '__main__':
    main()
