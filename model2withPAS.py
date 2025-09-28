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
    # 画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数计算出来
    # plt.plot(fpr,tpr,lw=1,alpha=0.3,label='ROC fold %d(area=%0.2f)'% (i,roc_auc))
    # i +=1
    # plt.plot([0,1],[0,1],linestyle='--',lw=2,color='r',label='Luck',alpha=.8)

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



from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Concatenate
from tensorflow.keras.layers import MultiHeadAttention
def GSRwithoe(shape1=None, filters1=128, filters2=64,filters3=32,ks1=3, ks2=5,ks3=7,filter2=23, size2=6, filter3=28, size3=8,
                 filter4=11, size4=5, lstm1=189, penalty=0.005, TRAINX=None, TRAIBY=None, validX=None,
                 validY=None, lr=None):
    # shape1=input_shape2 =(one_enac.shape[1],one_enac.shape[2])
    NCPDenac_input = Input(shape=shape1, name='NCPDenac_input')
    # 第一个门控卷积
    feature1_1 = Convolution1D(filters2, ks1, padding='same', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(penalty))(NCPDenac_input)
    feature1_1 = BatchNormalization()(feature1_1)
    feature1_1 = Activation('gelu')(feature1_1)

    feature1_2 = Convolution1D(filters2, ks1+6, padding='same', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(penalty))(NCPDenac_input)
    feature1_2 = BatchNormalization()(feature1_2)
    feature1_2 = Activation('gelu')(feature1_2)

    feature1_3 = Convolution1D(filters2, ks1+12, padding='same', kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(penalty))(NCPDenac_input)
    feature1_3 = BatchNormalization()(feature1_3)
    feature1_3 = Activation('gelu')(feature1_3)

    feature = Concatenate(axis=-1)([feature1_1, feature1_2, feature1_3])
    print(feature.shape)
    # (None,604,64)
    feature = MaxPooling1D(pool_size=2)(feature)
    feature = Dropout(0.42)(feature)
    # 应用门控卷积
    bi_lstm1 = Bidirectional(LSTM(64, return_sequences=True))(feature)
    #attention_out = MultiHeadAttention(num_heads=8, key_dim=16)(bi_lstm1, bi_lstm1)
    g_features1 = Dropout(0.5)(bi_lstm1)

    out_xin = Flatten()(g_features1)
    Y1 = Dense(512, activation='relu', name='before_dense',kernel_regularizer=regularizers.l2(penalty))(out_xin)
    Y1 = Dropout(0.5)(Y1)
    #Y2 = Dense(256, activation='relu', name='before_dense', kernel_regularizer=regularizers.l2(penalty))(Y1)
    output = Dense(1, activation='sigmoid')(Y1)

    model = Model(inputs=[NCPDenac_input], outputs=output)
    print(model.summary())
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Nadam(learning_rate=0.0003),
                  metrics=['accuracy'])
    # checkpointer = ModelCheckpoint(monitor='val_accuracy', verbose=1, save_best_only=True)
    lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=6, verbose=1)

    earlystopper = EarlyStopping(monitor='val_accuracy', patience=16, verbose=1)

    model.fit([x_train], y_train, epochs=200, batch_size=64,
              shuffle=True,
              callbacks=[earlystopper, lr_reduce],
              validation_data=([x_val], y_val), verbose=1)

    return model


# ENAC
enac = np.load(r'D:\我的代码\TIS\ENAC\dm-hAATAAA\train-ENAC.npy', allow_pickle=True)
#print(enac.shape)
enac = np.reshape(enac, ([enac.shape[0], 599, 4]))
print(enac.shape)
num = int(enac.shape[0] / 2)
negy = np.zeros(num)
posy = np.ones(num)
y_data = np.concatenate((negy, posy), axis=0)
enac = enac.astype(float)

path4 = r'D:\我的代码\TIS\NCPD\dm-hAATAAA\train-NCPD.npy'
NCPD = np.load(path4, allow_pickle=True)
#print(NCPD.shape)
NCPD_enac = np.concatenate((NCPD, enac), axis=1)
print(NCPD_enac.shape)
# (16950, 1208, 4)
NCPD_enac = NCPD_enac.astype(float)

input_shape2 = (NCPD_enac.shape[1], NCPD_enac.shape[2])

x_train, x_val, y_train, y_val = train_test_split(NCPD_enac, y_data, test_size=0.2, random_state=7)
models = GSRwithoe(shape1=input_shape2, )
# KF = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
pred = models.predict(x_val)
print(pred.shape)
# (3390,1)
accur = acc(y_val, pred)
models.save('./final_final/model/dm-hAATAAA/' + 'NCPD_enac卷积LSTMgelu激活' + str(accur) + '.h5')
print('model save')