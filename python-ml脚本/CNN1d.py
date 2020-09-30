# -*- coding: UTF-8 -*-

"""
搭建1维CNN
"""

import utils
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.utils import plot_model
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split


class CNN1D(object):
    def __init__(self, input_shape, num_classes, structure=0):
        self.history = None
        if structure == 0:
            self.model = Sequential()
            self.model.add(Reshape((input_shape//2, 2), input_shape=(input_shape, )))
            self.model.add(Conv1D(128, 7, activation='relu', padding='same',
                                  kernel_initializer='he_normal', input_shape=(input_shape//2, 2)))
            ## self.model.add(Conv1D(128, 10, activation='relu', padding='same',
            ##                       kernel_initializer='he_normal', input_shape=(600, 2)))
            self.model.add(MaxPooling1D(5))
            self.model.add(Conv1D(256, 7, activation='relu', padding='same', kernel_initializer='he_normal'))
            self.model.add(GlobalAveragePooling1D())
            self.model.add(Dropout(0.5))
            self.model.add(Dense(num_classes, activation='softmax'))
            self.model.compile(optimizer=keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        elif structure == 1:
            ts_input = keras.Input(shape=(input_shape,), name='ts')
            ts1 = Reshape((input_shape, 1))(ts_input)
            ts1 = Conv1D(64, 7, activation='relu')(ts1)
            ts1 = keras.layers.BatchNormalization()(ts1, training=False)
            ts1 = MaxPooling1D(3)(ts1)
            ts1 = Conv1D(128, 7, activation='relu')(ts1)
            ts1 = keras.layers.BatchNormalization()(ts1, training=False)
            ts1 = MaxPooling1D(3)(ts1)
            ts1 = Flatten()(ts1)

            fs_input = keras.Input(shape=(input_shape,), name='fs')
            fs1 = Reshape((input_shape, 1))(fs_input)
            fs1 = Conv1D(64, 7, activation='relu')(fs1)
            fs1 = keras.layers.BatchNormalization()(fs1, training=False)
            fs1 = MaxPooling1D(3)(fs1)
            fs1 = Conv1D(128, 7, activation='relu')(fs1)
            fs1 = keras.layers.BatchNormalization()(fs1, training=False)
            fs1 = MaxPooling1D(3)(fs1)
            fs1 = Flatten()(fs1)

            hidden = keras.layers.concatenate([ts1, fs1])
            hidden = Dense(128, activation='relu')(hidden)
            pred = Dense(7, activation='softmax')(hidden)
            self.model = keras.Model(inputs=[ts_input, fs_input], outputs=pred)
            self.model.compile(optimizer=keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        print(self.model.summary())

    def train_model(self, X_train, Y_train, X_test=None, Y_test=None, batch_size=128, epochs=50, structure=0):
        # self.__cnn1d_model()
        plot_model(self.model, to_file='../model/CNN1D_model.png')
        callbacks_list = [
            # keras.callbacks.ModelCheckpoint(
            #     filepath='../model/best_cnn_model.{epoch:02d}-{val_loss:.2f}.h5',
            #     monitor='val_loss', save_best_only=True),
            keras.callbacks.EarlyStopping(monitor='acc', patience=2)
        ]
        if structure == 0:
            self.history = self.model.fit(X_train, Y_train,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          callbacks=callbacks_list,
                                          class_weight='auto')
        elif structure == 1:
            train_a = X_train[:, 0:X_train.shape[1]//2]
            train_b = X_train[:, X_train.shape[1]//2:X_train.shape[1]]
            test_a = X_test[:, 0:X_train.shape[1]//2]
            test_b = X_test[:, X_train.shape[1]//2:X_train.shape[1]]
            print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
            self.history = self.model.fit({"ts": train_a, "fs": train_b}, Y_train, epochs=20,
                                          validation_data=({"ts": test_a, "fs": test_b}, Y_test))
        pass

    def predict(self, X_test, batch_size=128):
        return self.model.predict(X_test, batch_size=batch_size)

    def evaluate(self, X_test, Y_test, batch_size=128):
        return self.model.evaluate(X_test, Y_test, batch_size=batch_size)

    def save_CNN_model(self, name='../model/cnn1d.h5'):
        self.model.save(name)

    def load_CNN_model(self, name='../model/cnn1d.h5'):
        self.model = load_model(name)

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
    print('--- 1D CNN ---')
    cnn_model = CNN1D(input_shape=1200, num_classes=7)
    for cur_iter in range(iter_time):
        print('training iter:', cur_iter)
        # rus = RandomUnderSampler()
        for i in range(0, 29):
            utils.delete_data()
            begin, end = i, i + 1
            utils.simple_preprocessed(begin, end)
            samples, labels = utils.get_slice_data((120, 600), begin, end)
            X_train, Y_train = samples, labels
            ##X_train, X_test, Y_train, Y_test = train_test_split(samples, labels, shuffle=True, test_size=0.9)
            cnn_model.train_model(X_train, Y_train, batch_size=batch_size)
            cnn_model.visualize()
            cnn_model.save_CNN_model()
    return cnn_model


if __name__ == '__main__':
    main()
