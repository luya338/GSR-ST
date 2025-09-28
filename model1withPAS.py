import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Convolution1D, Softmax, Add, Lambda, MaxPooling1D, Dense, Dropout, Permute, multiply, Input, concatenate, BatchNormalization, Activation, Flatten, Bidirectional, LSTM, GRU
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import re
import scipy.io as scio
from sklearn.model_selection import train_test_split

import numpy as np
from keras import backend as K
import math
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, precision_recall_curve

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

file_path = r"E:\GSR-ST\GSR-ST\hAATAAA 训练集_last_hidden_states.npy"
print(f"文件路径: {file_path}")
print(f"文件是否存在: {os.path.exists(file_path)}")
X_data = np.load(file_path)



from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Concatenate
def GSRwithdnabert(shape1=None,  filters=64, ks=3, filter2=23, size2=6, filter3=28, size3=8,
                               filter4=11, size4=5, lstm1=189, penalty=0.005, TRAINX=None, TRAIBY=None, validX=None,
                               validY=None, lr=None):

    bert_input = Input(shape=shape1,name='bert_input')

    feature1_1 = Convolution1D(filters, ks, padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=regularizers.l2(penalty))(bert_input)
    feature1_1 = BatchNormalization()(feature1_1)
    feature1_1 = Activation('relu')(feature1_1)

    feature1_2 = Convolution1D(filters, ks + 6, padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=regularizers.l2(penalty))(bert_input)
    feature1_2 = BatchNormalization()(feature1_2)
    feature1_2 = Activation('relu')(feature1_2)

    feature1_3 = Convolution1D(filters, ks + 12, padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=regularizers.l2(penalty))(bert_input)
    feature1_3 = BatchNormalization()(feature1_3)
    feature1_3 = Activation('relu')(feature1_3)
    feature = Concatenate(axis=-1)([feature1_1, feature1_2, feature1_3])
    #print(feature.shape)
    feature = MaxPooling1D(pool_size=4)(feature)
    feature = Dropout(0.42)(feature)
    #print(feature.shape)

    bi_lstm1 = Bidirectional(LSTM(64, return_sequences=True))(feature)
    g_features1 = Dropout(0.58)(bi_lstm1)

    out_xin= Flatten()(g_features1)

    Y1 = Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(penalty))(out_xin)
    Y1 = Dropout(0.5)(Y1)
    Y3 = Dense(256, activation='relu', name='before_dense', kernel_regularizer=regularizers.l2(penalty))(Y1)

    output = Dense(2, activation='softmax',name='bert_output')(Y3)

    model = Model(inputs=[bert_input], outputs=output)
    print(model.summary())

    model.compile(loss={'bert_output': 'categorical_crossentropy'}, optimizer=tf.keras.optimizers.Nadam(learning_rate=0.0001), metrics=['accuracy'])


    lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.6, patience=6, verbose=1)
    earlystopper = EarlyStopping(monitor='val_accuracy', patience=21, verbose=1)
    def data_generator(w_train, y_train, batch_size):
        while True:
            for i in range(0, len(w_train), batch_size):
                yield w_train[i:i + batch_size], y_train[i:i + batch_size]

    model.fit(
        data_generator(w_train, y_train1, batch_size=64),
        epochs=200,
        steps_per_epoch = math.ceil(len(w_train) / 64),
        shuffle=True,
        callbacks=[earlystopper, lr_reduce],
        validation_data=([w_val], y_val1),
        verbose=1
    )
    return model


w_num = X_data.shape[0]
w_len = X_data.shape[1]
w_dim = X_data.shape[2]
input_shape1 = (w_len, w_dim)

num = int(w_num / 2)
negy = np.zeros(num)
posy = np.ones(num)

y_data = np.concatenate((negy, posy), axis=0)

from tensorflow.keras.utils import to_categorical
y_data=to_categorical(y_data)
w_train, w_val, y_train1, y_val1=train_test_split(X_data, y_data, test_size=0.2, random_state=7)

model1 = GSRwithdnabert(shape1=input_shape1     )
pred=model1.predict(w_val)
#print(pred.shape)

accur=acc(y_val1[:,1],pred[:,1])
model1.save('./final_final/model/allfruitflyhAATAAA/'  + 'bert卷积LSTMrelu激活' + str(accur) + '.h5')


print('model save')









