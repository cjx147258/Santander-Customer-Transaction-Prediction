# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 23:17:32 2019

@author: 61446
"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils


def drop_noisy(df):
    df_copy = df.copy()
    df_describe = df_copy.describe()
    for column in df.columns:
        mean = df_describe.loc['mean', column]
        std = df_describe.loc['std', column]
        minvalue = mean - 3*std
        maxvalue = mean + 3*std
        df_copy = df_copy[df_copy[column] >= minvalue]
        df_copy = df_copy[df_copy[column] <= maxvalue]
    return df_copy


model = Sequential()


data = pd.read_csv('data_st')
# data = drop_noisy(data)

data = np.array(data)
data = data[:, 1:]
print(data.shape)
np.random.shuffle(data)
np.random.shuffle(data)
print(data)
x_train = data[:, 1:]
y_train = data[:, :1].T
print(y_train.sum())
y_trainOneHot = np_utils.to_categorical(y_train)
model.add(Dense(units=200, activation='relu', input_dim=200))
# model.add(Dropout(0.2))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=50, activation='relu'))

model.add(Dense(units=2, activation='softmax'))


sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)
if input('load?') == '1':
    model.load_weights('model1')
conti = '1'
epoch = 10
batch = 10000
lr = 0.01
while conti == '1':
    conti = input("continue?")
    if conti != '1':
        break
    lr = input("new learning rate")
    epoch = input('epochs')
    batch = int(input("batch"))
    # sgd = SGD(lr=float(lr), decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='categorical_crossentropy', optimizer=sgd)
    model.fit(x_train, y_trainOneHot, batch_size=batch, epochs=int(epoch))
score = model.evaluate(x_train, y_trainOneHot, batch_size=batch)

result = model.predict(x_train, batch_size=batch, verbose=1)
print(result)
result_max = np.argmax(result, axis=1)
print(result_max)
result_bool = np.equal(result_max, y_train)
print(y_train)
print(result_bool)
# print(len(result))
print('测试集预测准确率：', result_bool.sum()/len(result))
save = input("save?")
if save == '1':
    model.save('model1')
