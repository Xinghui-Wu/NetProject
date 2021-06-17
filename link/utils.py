'''
Author: your name
Date: 2021-06-15 11:17:03
LastEditTime: 2021-06-15 15:54:36
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /NetProject/classification/utils.py
'''
import os
import numpy as np
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l2
from keras.models import Sequential


def sort_data(data_list):
    '''
    input: featur_extraction output: a data_list, the format of each element is (data,label)

    output: data array and label arrary
    '''
    x_list = []
    y_list = []
    for data in data_list:
        x_list.append(data[0])
        y_list.append(data[1])
    x_array = np.array(x_list)
    y_array = np.array(y_list)
    return x_array, y_array


def build_model(input_dim, class_num, weight_decay=0.0001, dense_units=128):
    model = Sequential()
    model.add(
        Dense(4 * dense_units,
              input_dim=input_dim,
              kernel_initializer='he_normal',
              kernel_regularizer=l2(weight_decay),
              activation='relu'))
    # model.add(Dense(4*dense_units, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),activation='relu'))
    # model.add(Dropout(0.25))
    # model.add(Dense(2*dense_units, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),activation='relu'))
    # model.add(Dense(dense_units, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),activation='relu'))
    model.add(
        Dense(dense_units,
              kernel_initializer='he_normal',
              kernel_regularizer=l2(weight_decay),
              activation='relu'))
    model.add(Dropout(0.25))
    model.add(
        Dense(class_num,
              kernel_initializer='he_normal',
              kernel_regularizer=l2(weight_decay)))
    model.add(Activation('softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.save('./classification/model.h5')
    return model


if __name__ == '__main__':
    build_model()