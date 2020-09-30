# -*- coding: UTF-8 -*-

"""
将多个片段的预测结果聚合
"""

import numpy as np
import utils
import matplotlib.pyplot as plt
import keras
from keras.utils import plot_model
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Activation
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D
from imblearn.under_sampling import RandomUnderSampler


class AggregationNN(object):
    def __init__(self, input_shape, num_classes):
        self.history = None
        self.model = Sequential()
        self.model.add(Reshape((120, 7), input_shape=(input_shape,)))
        self.model.add(Conv1D(64, 5, activation='relu', padding='same',
                              kernel_initializer='he_normal', input_shape=(120, 7)))
        # self.model.add(Conv1D(64, 5, activation='relu', padding='same', kernel_initializer='he_normal'))
        self.model.add(GlobalAveragePooling1D())
        self.model.add(Dropout(0.5))
        # self.model.add(Dense(units=10, kernel_initializer='he_normal', activation='relu'))
        self.model.add(Dense(units=10, kernel_initializer='he_normal', activation='relu'))
        self.model.add(Dense(units=num_classes, kernel_initializer='he_normal', activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    def train_model(self, X_train, Y_train, batch_size=128, epochs=10):
        plot_model(self.model, to_file='../model/aggrNN_model.png')
        callbacks_list = [
            # keras.callbacks.ModelCheckpoint(
            #     filepath='../model/best_aggr_model.{epoch:02d}-{val_loss:.2f}.h5',
            #     monitor='val_loss', save_best_only=True),
            keras.callbacks.EarlyStopping(monitor='acc', patience=2)
        ]
        self.history = self.model.fit(X_train, Y_train,
                                      batch_size=batch_size,
                                      epochs=epochs,
                                      callbacks=callbacks_list,
                                      class_weight='auto')
        pass

    def predict(self, X_test, batch_size=128):
        return self.model.predict(X_test, batch_size=batch_size)

    def evaluate(self, X_test, Y_test, batch_size=128):
        return self.model.evaluate(X_test, Y_test, batch_size=batch_size)

    def save_aggrNN_model(self):
        self.model.save('../model/aggrNN.h5')

    def load_aggrNN_model(self):
        self.model = load_model('../model/aggrNN.h5')

    # 可视化
    def visualize(self):
        # 绘制训练 & 验证的准确率值
        plt.figure()
        plt.plot(self.history.history['acc'])
        # plt.plot(self.history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        # 绘制训练 & 验证的损失值
        plt.figure()
        plt.plot(self.history.history['loss'])
        # plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()


def main(iter_time=1, batch_size=64):
    print('Aggregation NN')
    aggrNN = AggregationNN(input_shape=120*7, num_classes=7)
    for cur_iter in range(iter_time):
        print('training iter:', cur_iter)
        # rus = RandomUnderSampler()
        for i in range(0, 29):
            utils.delete_data()
            labels = np.load('../CNN_temp_data/' + str(i) + '_final_results.npy')  # 加载CNN1d.py得到的中间预测结果
            samples = np.load('../CNN_temp_data/' + str(i) + '_intermediate_results.npy')
            print(samples.shape, labels.shape)
            aggrNN.train_model(samples, labels, batch_size=batch_size)
            aggrNN.save_aggrNN_model()
    return aggrNN


if __name__ == '__main__':
    main()
