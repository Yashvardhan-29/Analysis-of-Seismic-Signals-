from re import X

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
from pathlib import Path

import scipy
from scipy import signal

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import itertools

import tensorflow as tf
from tensorflow.keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split


def plot_conf_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def make_model(samples,input_series=1):
    model = Sequential((
        Convolution1D(filters=64, kernel_size=8, activation='relu', input_shape=(samples, input_series)),
        MaxPooling1D(),
        Convolution1D(filters=128, kernel_size=5, activation='relu'),
        MaxPooling1D(),
        Convolution1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(),
        Flatten(),
        Dense(units=50, activation='relu'),
        Dense(units=3, activation='softmax'),
    ))
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(X, y):

    timeseries = X
    timeseries = timeseries.T
    t_samples = timeseries.shape[0]
    t_series = 1

    print('\n\nTimeseries ({} samples by {} series):'.format(t_samples, t_series))

    X = X.to_numpy().reshape((225, 5100, 1))
    y = y.to_numpy().reshape((225, 1))

    print('\nInput features:', X, '\n\nOutput labels:', y, sep='\n')

    model = make_model(samples=t_samples, input_series=t_series)
    print('\n\nModel with input size {}, output size {}'.format(model.input_shape, model.output_shape))
    model.summary()

    # training_data, testing_data = train_test_split(df, test_size=0.2, random_state=25)



    test_size = int(0.1 * timeseries.shape[1])
    X_train, X_test, y_train, y_test = X[:-test_size], X[-test_size:], y[:-test_size], y[-test_size:]
    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)
    # print(timeseries.shape)

    # X_train, X_test = train_test_split(X, test_size=0.2, random_state=25)
    # y_train, y_test = train_test_split(y, test_size=0.2, random_state=25)

    model.fit(X_train, y_train, epochs=50, batch_size=3, validation_split=0.2)

    pred = model.predict(X_test, batch_size=10)
    rounded_pred = np.argmax(pred, axis=-1)

    cm = confusion_matrix(y_true=y_test, y_pred=rounded_pred)

    cm_plot_labels = ['right', 'wrong']
    plot_conf_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

    # plot_confusion_matrix(model, X_test, y_test)
    # plt.show()

def data_preparation():
    # dirname = r'C:\Users\bansa\Documents\Python\csio_rw\22 Nov\Walking'
    dirname = r'C:\Users\DELL\Desktop\Isens_backyard testing\Sensor_parallel to trench\22_Nov\Sensorplace at edge of trench\Subject_Anuj\Walk\lvm'

    paths = Path(dirname).glob('**/*.lvm', )

    df = pd.DataFrame()

    for path in paths:
        a = pd.read_table(path, header=None)
        a.drop([0, 2, 5, 6, 7, 8], axis=1, inplace=True)
        a = pd.DataFrame(signal.detrend(a))
        #     print(a.head())
        df = pd.concat([df, a], axis=1)

    df_max_scaled = df.copy()

    for column in df_max_scaled.columns:
        df_max_scaled[column] = df_max_scaled[column] / df_max_scaled[column].abs().max()

    # df_max_scaled

    bools = df_max_scaled.isnull().any(axis=1)

    # bools = bools.index[~bools.shift(fill_value=True) & bools]

    idx = 0
    for b in bools:
        if b == True:
            break
        idx += 1

    # idx

    t_df = df_max_scaled.drop(df_max_scaled.index[idx:])
    df_walk = t_df.T
    df_walk = df_walk.reset_index().drop('index', axis=1)

    df1 = df_walk.iloc[:, :5100]
    df2 = df_walk.iloc[:, 5100:10200]
    df3 = df_walk.iloc[:, 10200:15300]
    df4 = df_walk.iloc[:, 15300:20400]

    df4.columns = df3.columns = df2.columns = df1.columns

    final_df_walk = pd.concat([df1, df2, df3, df4], axis=0, ignore_index=True)
    final_df_walk['Activity'] = 0

    # dirname = r'C:\Users\bansa\Documents\Python\csio_rw\22 Nov\Jogging'
    dirname = r'C:\Users\DELL\Desktop\Isens_backyard testing\Sensor_parallel to trench\22_Nov\Sensorplace at edge of trench\Subject_Anuj\Jog\lvm'

    paths = Path(dirname).glob('**/*.lvm', )

    df = pd.DataFrame()

    for path in paths:
        a = pd.read_table(path, header=None)
        a.drop([0, 2, 5, 6, 7, 8], axis=1, inplace=True)
        a = pd.DataFrame(signal.detrend(a))
        #     print(a.head())
        df = pd.concat([df, a], axis=1)

    df_max_scaled = df.copy()

    for column in df_max_scaled.columns:
        df_max_scaled[column] = df_max_scaled[column] / df_max_scaled[column].abs().max()

    # df_max_scaled

    bools = df_max_scaled.isnull().any(axis=1)

    # bools = bools.index[~bools.shift(fill_value=True) & bools]

    idx = 0
    for b in bools:
        if b == True:
            break
        idx += 1

    # idx

    t_df = df_max_scaled.drop(df_max_scaled.index[idx:])
    df_jog = t_df.T
    df_jog = df_jog.reset_index().drop('index', axis=1)
    df1 = df_walk.iloc[:, :5100]
    df2 = df_walk.iloc[:, 5100:10200]

    df2.columns = df1.columns

    final_df_jog = pd.concat([df1, df2], axis=0, ignore_index=True)
    final_df_jog['Activity'] = 1

    dirname = r'C:\Users\DELL\Desktop\Isens_backyard testing\Sensor_parallel to trench\22_Nov\Sensorplace at edge of trench\Subject_Anuj\Hammering\lvm'

    paths = Path(dirname).glob('**/*.lvm', )

    df = pd.DataFrame()

    for path in paths:
        a = pd.read_table(path, header=None)
        a.drop([0, 2, 5, 6, 7, 8], axis=1, inplace=True)
        a = pd.DataFrame(signal.detrend(a))
        #     print(a.head())
        df = pd.concat([df, a], axis=1)

    df_max_scaled = df.copy()

    for column in df_max_scaled.columns:
        df_max_scaled[column] = df_max_scaled[column] / df_max_scaled[column].abs().max()

    # df_max_scaled

    bools = df_max_scaled.isnull().any(axis=1)

    # bools = bools.index[~bools.shift(fill_value=True) & bools]

    idx = 0
    for b in bools:
        if b == True:
            break
        idx += 1

    # idx

    t_df = df_max_scaled.drop(df_max_scaled.index[idx:])

    df_hammer = t_df.T
    df_hammer = df_hammer.reset_index().drop('index', axis=1)

    final_df_hammer = df_hammer.iloc[:, :5100]
    final_df_hammer['Activity'] = 2

    df = pd.DataFrame()
    df = pd.concat([final_df_walk, final_df_jog, final_df_hammer], axis=0)

    df = df.sample(frac=1)
    df = df.reset_index().drop('index', axis=1)
    # df.shape

    print(df)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return X,y

def main():

    samples, labels = data_preparation()
    evaluate_model(X=samples, y=labels)

if __name__ == '__main__':
    main()
