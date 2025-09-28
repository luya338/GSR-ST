import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.keras.layers import Bidirectional, GRU, MaxPooling1D, Dropout
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Convolution1D, Softmax, Add, Lambda, MaxPooling1D, Dense, Dropout, \
    Permute, multiply, Input, concatenate, BatchNormalization, Activation, Flatten, Bidirectional, LSTM, GRU
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Attention

from attention import *
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


def Twoclassfy_evalu(y_test, y_predict1):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    FP_index = []
    FN_index = []
    aucs = []
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

    Sn = TP / (TP + FN)
    Sp = TN / (FP + TN)
    Acc = (TP + TN) / (TP + FP + TN + FN)
    Mcc = (TP * TN - FP * FN) / (math.sqrt(TP + FP) * math.sqrt(TP + FN) * math.sqrt(TN + FP) * math.sqrt(TN + FN))
    Precision = TP / (TP + FP)
    F1_score = (2 * Precision * Sn) / (Precision + Sn)
    fpr, tpr, thresholds = roc_curve(y_test, y_predict1)
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    print('TP', TP)
    print('FP', FP)
    print('FN', FN)
    print('TN', TN)
    print('Sn', Sn)
    print('Sp', Sp)
    print('ACC', Acc)
    print('Mcc', Mcc)
    print('Precision', Precision)
    print('F1_score', F1_score)
    print('AUC', aucs)
    return TP, FP, FN, TN, Sn, Sp, Acc, Mcc, Precision, F1_score, aucs


def acc(y_test, y_predict1):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    FP_index = []
    FN_index = []
    aucs = []
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

from tensorflow.keras import backend as K




from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Concatenate
import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
import math

from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score

def GSRwithoe(shape1=None, filters1=128, filters2=64,filters3=32,ks1=3, ks2=5,ks3=7,filter2=23, size2=6, filter3=28, size3=8,
                 filter4=11, size4=5, lstm1=189, penalty=0.005, TRAINX=None, TRAIBY=None, validX=None,
                 validY=None, lr=None):
    NCPDenac_input = Input(shape=shape1, name='NCPDenac_input')
    feature1_1 = Convolution1D(filters2, ks1, padding='same', kernel_initializer='lecun_normal',kernel_regularizer=regularizers.l2(penalty))(NCPDenac_input)
    feature1_1 = BatchNormalization()(feature1_1)
    feature1_1 = Activation('selu')(feature1_1)

    feature1_2 = Convolution1D(filters2, ks1+6, padding='same', kernel_initializer='lecun_normal',kernel_regularizer=regularizers.l2(penalty))(NCPDenac_input)
    feature1_2 = BatchNormalization()(feature1_2)
    feature1_2 = Activation('selu')(feature1_2)

    feature1_3 = Convolution1D(filters2, ks1+12, padding='same', kernel_initializer='lecun_normal',kernel_regularizer=regularizers.l2(penalty))(NCPDenac_input)
    feature1_3 = BatchNormalization()(feature1_3)
    feature1_3 = Activation('selu')(feature1_3)

    feature = Concatenate(axis=-1)([feature1_1, feature1_2, feature1_3])
    #print(feature.shape)
    feature = MaxPooling1D(pool_size=2)(feature)
    feature = Dropout(0.42)(feature)
    bi_lstm1 = Bidirectional(LSTM(64, return_sequences=True))(feature)
    g_features1 = Dropout(0.58)(bi_lstm1)

    out_xin = Flatten()(g_features1)
    Y1 = Dense(512, activation='relu', name='before_dense',kernel_regularizer=regularizers.l2(penalty))(out_xin)
    Y1 = Dropout(0.5)(Y1)
    output = Dense(1, activation='sigmoid')(Y1)
    model = Model(inputs=[NCPDenac_input], outputs=output)
    print(model.summary())
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Nadam(learning_rate=0.0003),
                  metrics=['accuracy'])
    return model

# ENAC
enac = np.load(r'D:\我的代码\启动子\train_data\S.Typhimurium\ENAC.npy', allow_pickle=True)
enac = np.reshape(enac, ([enac.shape[0], 77, 4]))  # 形状: (16950, 602, 4)
enac = enac.astype(float)

num = int(enac.shape[0] / 2)
negy = np.zeros(num)
posy = np.ones(num)
y_data = np.concatenate((negy, posy), axis=0)

path4 = r'D:\我的代码\启动子\train_data\S.Typhimurium\NCPD.npy'
NCPD = np.load(path4, allow_pickle=True)
NCPD_enac = np.concatenate((NCPD, enac), axis=1)
NCPD_enac = NCPD_enac.astype(float)
input_shape2 = (NCPD_enac.shape[1], NCPD_enac.shape[2])

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)

fold = 1
accuracies = []

for train_index, val_index in kf.split(NCPD_enac, y_data):
    print(f'第{fold}折...')
    x_train, x_val = NCPD_enac[train_index], NCPD_enac[val_index]
    y_train, y_val = y_data[train_index], y_data[val_index]
    model = GSRwithoe(shape1=input_shape2)

    earlystopper = EarlyStopping(monitor='val_accuracy', patience=16, verbose=1)
    lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=6, verbose=1)

    model.fit(x_train, y_train, epochs=200, batch_size=64,
              shuffle=True,
              validation_data=(x_val, y_val),
              callbacks=[earlystopper, lr_reduce],
              verbose=1)

    pred = model.predict(x_val)
    print(pred.shape)
    accur = acc(y_val, pred)
    accuracies.append(accur)

    model_save_dir = './B.amyloliquefaciens/'
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    model_path = f'{model_save_dir}/OE 卷积64 LSTM selu激活 fold_{fold}_{accur:.4f}.h5'
    model.save(model_path)
    print(f'Model for fold {fold} saved at {model_path}')

    fold += 1

avg_accuracy = np.mean(accuracies)
print(f'十折交叉验证的平均准确率: {avg_accuracy}')