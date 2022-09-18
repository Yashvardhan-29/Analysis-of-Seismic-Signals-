import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
from pathlib import Path

import scipy
from scipy import signal
import scipy.io

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import itertools

import tensorflow as tf
from tensorflow.keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten, BatchNormalization, Dropout
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
        Convolution1D(filters=512, kernel_size=3, activation='relu', input_shape=(samples, input_series)),
        MaxPooling1D(),
        Convolution1D(filters=256, kernel_size=3, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.001)),
        BatchNormalization(),
        # MaxPooling1D(pool_size=(3,), strides=2, padding='same'),
        Convolution1D(filters=128, kernel_size=3, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.001)),
        BatchNormalization(),
        # MaxPooling1D(pool_size=(3,), strides=2, padding='same'),
        Convolution1D(filters=64, kernel_size=3, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.001)),
        BatchNormalization(),
        # MaxPooling1D(pool_size=(3,), strides=2, padding='same'),
        Convolution1D(filters=32, kernel_size=3, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.001)),
        BatchNormalization(),
        # MaxPooling1D(pool_size=(3,), strides=2, padding='same'),
        # Dropout(0.5),
        Flatten(),
        Dense(units=16, activation='relu'),
        Dense(units=1, activation='sigmoid'),
    ))
    opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
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
    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)

    model_history = model.fit(X_train, y_train, epochs=40, batch_size=2, validation_split=0.21)

    results = model.evaluate(X_test, y_test, batch_size=1)
    print("test loss, test acc:", results)

    pred = model.predict(X_test)

    # print(type(pred))
    # print(np.shape(pred))
    print(pred) 

    rounded_pred = np.round_(pred)

    # print(rounded_pred)
    cm = confusion_matrix(y_true=y_test, y_pred=rounded_pred)

    cm_plot_labels = ['right', 'wrong']
    plot_conf_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

    # plot_confusion_matrix(model, X_test, y_test)
    # plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(np.sqrt(model_history.history['loss']), 'r', label='train')
    ax.plot(np.sqrt(model_history.history['val_loss']), 'b', label='val')
    ax.set_xlabel(r'Epoch', fontsize=20)
    ax.set_ylabel(r'Loss', fontsize=20)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.tick_params(labelsize=20)


    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(np.sqrt(model_history.history['accuracy']), 'r', label='train')
    ax.plot(np.sqrt(model_history.history['val_accuracy']), 'b', label='val')
    ax.set_xlabel(r'Epoch', fontsize=20)
    ax.set_ylabel(r'Accuracy', fontsize=20)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=20)
    plt.show()

def data_preparation():
    path = r'C:\Users\DELL\Desktop\Seismic Data\Walk'
    dir_list = os.listdir(path)

    df = pd.DataFrame()

    for file in dir_list:
        mat = scipy.io.loadmat(path + '\\' + file)
        # print(m.keys())
        a = mat['Data1_SNG20DX10Hz1']
        a = pd.DataFrame((a - np.mean(a)) / 20)
        df = pd.concat([df, a], axis=1)

    df = df.fillna(0)
    df = df.T.reset_index().drop('index', axis=1).T

    ts = np.empty([0, 250])

    for j in range(df.shape[1]):

        t = df[j]

        l = []
        th1 = 0.0000075
        th2 = -0.0000075
        for i in range(len(t)):
            if t[i] > th1 or t[i] < th2:
                l.append(i)
        # print(l)

        lf = []
        lf.append(l[0])
        for i in range(len(l) - 1):
            if l[i] + 150 < l[i + 1]:
                lf.append(l[i + 1])
        # print(lf)

        for i in range(len(lf)):
            ts = np.vstack([ts, t[lf[i] - 100:lf[i] + 150]])
            # break
        # break

    t1 = pd.DataFrame(ts)
    t1['Activity'] = 0

    path = r'C:\Users\DELL\Desktop\Seismic Data\Run'
    dir_list = os.listdir(path)

    df = pd.DataFrame()

    for file in dir_list:
        mat = scipy.io.loadmat(path + '\\' + file)
        # print(m.keys())
        b = mat['Data1_SNG20DX10Hz1']
        b = pd.DataFrame((b - np.mean(b)) / 20)
        df = pd.concat([df, b], axis=1)

    df = df.fillna(0)
    df = df.T.reset_index().drop('index', axis=1).T

    ts = np.empty([0, 250])

    for j in range(df.shape[1]):

        t = df[j]

        l = []
        th1 = 0.00002
        th2 = -0.00002
        for i in range(len(t)):
            if t[i] > th1 or t[i] < th2:
                l.append(i)
        # print(l)

        lf = []
        lf.append(l[0])
        for i in range(len(l) - 1):
            if l[i] + 150 < l[i + 1]:
                lf.append(l[i + 1])
        # print(lf)

        for i in range(len(lf)):
            ts = np.vstack([ts, t[lf[i] - 100:lf[i] + 150]])
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
