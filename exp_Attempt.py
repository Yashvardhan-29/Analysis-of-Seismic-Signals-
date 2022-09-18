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
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import itertools

import tensorflow as tf
from tensorflow.keras import layers
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

path = r'C:\Users\bansa\Documents\Python\csio_rw\Data\Seismic Data\Walk'
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

path = r'C:\Users\bansa\Documents\Python\csio_rw\Data\Seismic Data\Run'
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

x = df.iloc[:, :-1]
y = df.iloc[:, -1]

visible = tf.keras.Input(shape=(250,1))
l1 = layers.Conv1D(filters=4, kernel_size=(9,), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.001))(visible)
l2 = layers.BatchNormalization()(l1)
# #l5 = layers.MaxPool1D(pool_size=(3,), strides=2, padding='same')(l2)
# l3 = layers.Conv1D(filters=4, kernel_size=(5,), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.001))(l2)
# l4 = layers.BatchNormalization()(l3)
# #l5 = layers.MaxPool1D(pool_size=(3,), strides=2, padding='same')(l4)
# l13 = layers.Conv1D(filters=128, kernel_size=(3,), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.001))(l4)
# l14 = layers.BatchNormalization()(l13)
# #l15 = layers.MaxPool1D(pool_size=(3,), strides=2, padding='same')(l14)
# l23 = layers.Conv1D(filters=64, kernel_size=(3,), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.001))(l14)
# l24 = layers.BatchNormalization()(l23)
# #l25 = layers.MaxPool1D(pool_size=(3,), strides=2, padding='same')(l24)
# l33 = layers.Conv1D(filters=32, kernel_size=(3,), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.001))(l24)
# l34 = layers.BatchNormalization()(l33)
# #l35 = layers.MaxPool1D(pool_size=(3,), strides=2, padding='same')(l34)
# #l6 = layers.Dropout(0.5)(l35)
l7 = layers.Flatten()(l2)
l8 = layers.Dense(units = 256, activation=tf.keras.layers.LeakyReLU(alpha=0.001))(l7)
l9 = layers.Dense(units = 128, activation=tf.keras.layers.LeakyReLU(alpha=0.001))(l8)
l10 = layers.Dense(units = 64, activation=tf.keras.layers.LeakyReLU(alpha=0.001))(l9)
output = layers.Dense(units = 1, activation='sigmoid')(l10)
from keras.models import Model
model = Model(inputs=visible, outputs=output)

c1 = model.compile(loss = 'binary_crossentropy', optimizer = tf.keras.optimizers.Adam(), metrics = ['accuracy'])
model.summary()

test_size = int(0.1 * x.shape[0])
print(test_size)
x_train, x_test, y_train, y_test = x[:-test_size], x[-test_size:], y[:-test_size], y[-test_size:]

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

model_history = model.fit(x_train,y_train,batch_size=5,epochs=20,validation_split=0.2)

results = model.evaluate(x_test, y_test, batch_size=1)
print("test loss, test acc:", results)

pred = model.predict(x_test)
rounded_pred = np.round_(pred)

# print(rounded_pred)
cm = confusion_matrix(y_true=y_test, y_pred=rounded_pred)

cm_plot_labels = ['Walking', 'Running']
plot_conf_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot((model_history.history['loss']), 'r', label='train')
ax.plot((model_history.history['val_loss']), 'b', label='val')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Loss', fontsize=20)
ax.set_ylim(0, 1.5)
ax.legend()
ax.tick_params(labelsize=20)


fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot((model_history.history['accuracy']), 'r', label='train')
ax.plot((model_history.history['val_accuracy']), 'b', label='val')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Accuracy', fontsize=20)
ax.legend()
ax.set_ylim(0, 1.5)
ax.tick_params(labelsize=20)
plt.show()


print('Precision: %.3f' % precision_score(y_test, rounded_pred))

print('Recall: %.3f' % recall_score(y_test, rounded_pred))

print('Accuracy: %.3f' % accuracy_score(y_test, rounded_pred))

print('F1 Score: %.3f' % f1_score(y_test, rounded_pred))
