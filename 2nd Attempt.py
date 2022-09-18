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


def make_model(samples, input_series=1):
    model = Sequential((
        Convolution1D(filters=64, kernel_size=8, activation='relu', input_shape=(samples, input_series)),
        MaxPooling1D(),
        Convolution1D(filters=128, kernel_size=5, activation='relu'),
        MaxPooling1D(),
        Convolution1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(),
        Flatten(),
        Dense(units=50, activation='relu'),
        Dense(units=1, activation='sigmoid'),
    ))
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
    return model


def evaluate_model(X, y):
    ts = X
    t_samples = ts.shape[1]
    t_series = ts.shape[0]

    print('\n\nTimeseries ({} samples by {} series):'.format(t_samples, t_series))

    print('\nInput features:', X, '\n\nOutput labels:', y, sep='\n')
    model = make_model(samples=t_samples, input_series=1)

    print('\n\nModel with input size {}, output size {}'.format(model.input_shape, model.output_shape))
    model.summary()

    test_size = int(0.1 * ts.shape[0])
    X_train, X_test, y_train, y_test = X[:-test_size], X[-test_size:], y[:-test_size], y[-test_size:]

    model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.2)

    pred = model.predict(X_test, batch_size=10)
    rounded_pred = np.argmax(pred, axis=-1)

    cm = confusion_matrix(y_true=y_test, y_pred=rounded_pred)

    cm_plot_labels = ['right', 'wrong']
    plot_conf_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

    # plot_confusion_matrix(model, X_test, y_test)
    # plt.show()


def data_preparation():
    dirname = r'C:\Users\DELL\Desktop\Jog_lvm'
    paths = Path(dirname).glob('**/*.lvm', )

    df = pd.DataFrame()

    for path in paths:
        s = pd.read_table(path, header=None)
        s.drop([0, 2, 5, 6, 7, 8], axis=1, inplace=True)
        s = pd.DataFrame(signal.detrend(s) / 85.5)
        #     print(a.head())
        df = pd.concat([df, s], axis=1)

    df = df.fillna(0)
    df = df.T.reset_index().drop('index', axis=1).T

    ts = np.empty([0, 250])

    for j in range(df.shape[1]):

        t = df[j]
        # print(type(t))

        l = []
        th = 0.00001025 / 2
        for i in range(len(t)):
            if t[i] > th:
                l.append(i)
        # print(l)

        lf = []
        lf.append(l[0])
        for i in range(len(l) - 1):
            if l[i] + 150 < l[i + 1]:
                lf.append(l[i + 1])
        # print(lf)

        for i in range(len(lf)):
            # ts = pd.concat([ts, t[lf[i]-100:lf[i]+150]], ignore_index=True, axis=0)
            # ts = ts.append(t[lf[i]-100:lf[i]+150], ignore_index=True)
            ts = np.vstack([ts, t[lf[i] - 100:lf[i] + 150]])
            # print(np.shape(ts))
            # print(t[lf[i]-100:lf[i]+150])
            # break
        # break

    t1 = pd.DataFrame(ts)
    t1['Activity'] = 0

    dirname = r'C:\Users\DELL\Desktop\Ham_lvm'
    paths = Path(dirname).glob('**/*.lvm', )

    df = pd.DataFrame()

    for path in paths:
        s = pd.read_table(path, header=None)
        s.drop([0, 2, 5, 6, 7, 8], axis=1, inplace=True)
        s = pd.DataFrame(signal.detrend(s) / 85.5)
        #     print(a.head())
        df = pd.concat([df, s], axis=1)

    df = df.fillna(0)
    df = df.T.reset_index().drop('index', axis=1).T

    ts = np.empty([0, 250])

    for j in range(df.shape[1]):

        t = df[j]
        # print(type(t))

        l = []
        th = 0.0000205
        for i in range(len(t)):
            if t[i] > th:
                l.append(i)
        # print(l)

        lf = []
        lf.append(l[0])
        for i in range(len(l) - 1):
            if l[i] + 150 < l[i + 1]:
                lf.append(l[i + 1])
        # print(lf)

        for i in range(len(lf)):
            # ts = pd.concat([ts, t[lf[i]-100:lf[i]+150]], ignore_index=True, axis=0)
            # ts = ts.append(t[lf[i]-100:lf[i]+150], ignore_index=True)
            ts = np.vstack([ts, t[lf[i] - 100:lf[i] + 150]])
            # print(np.shape(ts))
            # print(t[lf[i]-100:lf[i]+150])
            # break
        # break

    t2 = pd.DataFrame(ts)
    t2['Activity'] = 1

    df = pd.DataFrame()
    df = pd.concat([t1, t2], axis=0)
    df = df.sample(frac=1)
    df = df.reset_index().drop('index', axis=1)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return X, y


def main():
    samples, labels = data_preparation()
    evaluate_model(X=samples, y=labels)


if __name__ == '__main__':
    main()
