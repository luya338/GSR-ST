import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from untils import GSRwithbert

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Convolution1D, Softmax, Add, Lambda, MaxPooling1D, Dense, Dropout, Permute, multiply, Input, concatenate, BatchNormalization, Activation, Flatten, Bidirectional, LSTM, GRU
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import math
from nltk import bigrams, trigrams
from Bio import SeqIO
from gensim.models import Word2Vec
from bayes_opt import BayesianOptimization



from keras_pos_embd import PositionEmbedding

import re
import scipy.io as scio
from sklearn.model_selection import train_test_split

import numpy as np
from keras import backend as K
import math
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, precision_recall_curve
# 确认TensorFlow可以使用GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Using CPU instead.")
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import numpy as np
import math
import os


def Twoclassfy_evalu(y_test, y_predict1):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    FP_index = []
    FN_index = []
    aucs = []
    for i in range(len(y_test)):
        if y_predict1[i]> 0.5 and y_test[i] == 1:
            TP += 1
        if y_predict1[i] > 0.5 and y_test[i] == 0:
            FP += 1
            FP_index.append(i)
        if y_predict1[i]< 0.5 and y_test[i] == 1:
            FN += 1
            FN_index.append(i)
        if y_predict1[i] < 0.5 and y_test[i] == 0:
            TN += 1

    Sn = TP / (TP + FN)
    Sp = TN / (FP + TN)
    Acc = (TP + TN) / (TP + FP + TN + FN)
    if TP==0 or TN==0 or FN==0 or FP==0:
        print('TP==0 or TN==0 or FN==0 or FP==0')
        return Sn,Sp,Acc
    else:
        Mcc = (TP * TN - FP * FN) / (math.sqrt(TP + FP) * math.sqrt(TP + FN) * math.sqrt(TN + FP) * math.sqrt(TN + FN))
        Precision = TP / (TP + FP)
        F1_score = (2 * Precision * Sn) / (Precision + Sn)
        fpr,tpr,thresholds = roc_curve(y_test,y_predict1)
        roc_auc=auc(fpr,tpr)
        aucs.append(roc_auc)
        print('TP',TP)
        print('FP',FP)
        print('FN', FN)
        print('TN', TN)

        print('Sn', Sn)
        print('Sp', Sp)
        print('ACC', Acc)
        print('Mcc', Mcc)
        print('Precision',  Precision)
        print('F1_score', F1_score)
        print('AUC', aucs)
        return  TP, FP, FN, TN, Sn, Sp, Acc, Mcc, Precision, F1_score, aucs
def acc(y_test,y_predict1):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    FP_index = []
    FN_index = []
    for i in range(len(y_test)):
        if y_predict1[i] > 0.5 and y_test[i] == 1:
            TP += 1
        if y_predict1[i] > 0.5 and y_test[i] == 0:
            FP += 1
            FP_index.append(i)
        if y_predict1[i] < 0.5 and y_test[i] == 1:
            FN += 1
            FN_index.append(i)
        if y_predict1[i] < 0.5 and y_test[i] == 0:
            TN += 1
    Acc = (TP + TN) / (TP + FP + TN + FN)
    return Acc

def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.where(x > 0.0, x, alpha * tf.nn.elu(x))

import numpy as np

import os

file_path = r"E:\GSR-ST\GSR-ST\processed_data\Promoter_data\train_data\B.amyloliquefaciens\total_last_hidden_states.npy"
print(f"文件路径: {file_path}")
print(f"文件是否存在: {os.path.exists(file_path)}")
X_data = np.load(file_path)
#print(X_data.shape)

w_num = X_data.shape[0]
w_len = X_data.shape[1]
w_dim = X_data.shape[2]
input_shape1 = (w_len, w_dim)

num = int(w_num / 2)
negy = np.zeros(num)
posy = np.ones(num)
y_data = np.concatenate((negy, posy), axis=0)
y_data = to_categorical(y_data)

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold = 1
accuracies = []

for train_index, val_index in kf.split(X_data, y_data.argmax(axis=1)):
    print(f"Training fold {fold}...")
    w_train, w_val = X_data[train_index], X_data[val_index]
    y_train1, y_val1 = y_data[train_index], y_data[val_index]
    model = GSRwithbert(shape1=input_shape1)

    def data_generator(w_train, y_train, batch_size):
        while True:
            for i in range(0, len(w_train), batch_size):
                yield w_train[i:i + batch_size], y_train[i:i + batch_size]


    lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.6, patience=6, verbose=1)
    earlystopper = EarlyStopping(monitor='val_accuracy', patience=21, verbose=1)

    model.fit(
        data_generator(w_train, y_train1, batch_size=64),
        epochs=200,
        steps_per_epoch=math.ceil(len(w_train) / 64),
        shuffle=True,
        callbacks=[earlystopper, lr_reduce],
        validation_data=(w_val, y_val1),
        verbose=1
    )

    pred = model.predict(w_val)
    print(pred.shape)

    accur = acc(y_val1[:, 1], pred[:, 1])
    accuracies.append(accur)

    model_save_dir = './B.amyloliquefaciens/'
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    model_path = f'{model_save_dir}/bert 卷积64 LSTM selu激活 fold_{fold}_{accur:.4f}.h5'
    model.save(model_path)
    print(f'Model for fold {fold} saved at {model_path}')

    fold += 1


print("Cross-validation complete")
print(f"Mean accuracy: {np.mean(accuracies):.4f}, Std: {np.std(accuracies):.4f}")