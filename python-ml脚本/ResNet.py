# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from keras.models import load_model, Model
import keras
from keras import layers as ll
from keras.layers import Dense, Activation, Input
from keras.models import Sequential
from keras.layers.core import Flatten, Reshape
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D, AveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model
import utils
from sklearn.model_selection import train_test_split


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = BatchNormalization(name=bn_name_base + '2a')(input_tensor)
    x = Activation('relu')(x)
    x = Conv1D(filters1, 1, name=conv_name_base + '2a')(x)

    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = Conv1D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)

    x = BatchNormalization(name=bn_name_base + '2c')(x)
    x = Conv1D(filters3, 1, name=conv_name_base + '2c')(x)

    x = ll.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=2):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = BatchNormalization(name=bn_name_base + '2a')(input_tensor)
    x = Activation('relu')(x)
    x = Conv1D(filters1, 1, strides=strides, name=conv_name_base + '2a')(x)

    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = Conv1D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)

    x = BatchNormalization(name=bn_name_base + '2c')(x)
    x = Conv1D(filters3, 1, name=conv_name_base + '2c')(x)

    shortcut = BatchNormalization(name=bn_name_base + '1')(input_tensor)
    shortcut = Conv1D(filters3, 1, strides=strides, name=conv_name_base + '1')(shortcut)

    x = ll.add([x, shortcut])
    x = Activation('relu')(x)
    return x


class ResNet(object):
    def __init__(self, input_shape, classes=7):
        x_input = Input(input_shape)
        x = Reshape((600, 2), input_shape=(input_shape,))(x_input)
        x = Conv1D(64, 7, strides=2, padding='same', name='conv1')(x)
        x = BatchNormalization(name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(3, strides=2)(x)

        x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=1)
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        # x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        # x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        # x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        # x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        x = AveragePooling1D(classes, name='avg_pool')(x)

        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc'+str(classes))(x)
        self.model = Model(inputs=x_input, outputs=x, name='ResNet1D')
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.history = None

    def train_model(self, X_train, Y_train, batch_size=128, epochs=1):
        plot_model(self.model, to_file='../model/ResNet_model.png')
        callbacks_list = [
            # keras.callbacks.ModelCheckpoint(
            #     filepath='../model/best_resnet_model.{epoch:02d}-{val_loss:.2f}.h5',
            #     monitor='val_loss', save_best_only=True),
            keras.callbacks.EarlyStopping(monitor='acc', patience=2)
        ]
        self.history = self.model.fit(X_train, Y_train,
                                      batch_size=batch_size,
                                      epochs=epochs,
                                      callbacks=callbacks_list)

        pass

    def predict(self, X_test, batch_size=128):
        return self.model.predict(X_test, batch_size=batch_size)

    def evaluate(self, X_test, Y_test, batch_size=128):
        return self.model.evaluate(X_test, Y_test, batch_size=batch_size)

    def save_CNN_model(self):
        self.model.save('../model/resnet.h5')

    def load_CNN_model(self):
        self.model = load_model('../model/resnet.h5')

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
    print('--- ResNet ---')
    resnet_model = ResNet(input_shape=(1200, ), classes=7)
    for cur_iter in range(iter_time):
        print('training iter:', cur_iter)
        for i in range(0, 29):
            utils.delete_data()
            begin, end = i, i + 1
            utils.simple_preprocessed(begin, end)
            samples, labels = utils.get_slice_data((120, 600), begin, end)
            ## X_train, Y_train = samples, labels
            if i == 3:
                test_size = 0.9
            else:
                test_size = 0.95
            X_train, X_test, Y_train, Y_test = train_test_split(samples, labels, shuffle=True, test_size=test_size)
            resnet_model.train_model(X_train, Y_train, batch_size=batch_size)
            ## resnet_model.visualize()
            resnet_model.save_CNN_model()
    return resnet_model


if __name__ == '__main__':
    main(iter_time=5)
