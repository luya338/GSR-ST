import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Convolution1D, Softmax, Add, Lambda, MaxPooling1D, Dense, Dropout, Permute, multiply, Input, concatenate, BatchNormalization, Activation, Flatten, Bidirectional, LSTM, GRU
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

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
import  xgboost as xgb
from keras import backend as K

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
    if TP==0 ==0 or TN==0 or FP==0:
        print('TP==0 or FN==0 or TN==0 or FP==0')
        return Sn,Sp,Acc
    else:
        Mcc = (TP * TN - FP * FN) / (math.sqrt(TP + FP) * math.sqrt(TP + FN) * math.sqrt(TN + FP) * math.sqrt(TN + FN))
        Precision = TP / (TP + FP)
        F1_score = (2 * Precision * Sn) / (Precision + Sn)
        fpr,tpr,thresholds = roc_curve(y_test,y_predict1)
        roc_auc=auc(fpr,tpr)
        precision, recall, thresholds = precision_recall_curve(y_test,y_predict1)
        auprc = auc(recall, precision)
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
        print('auprc',auprc)
        return  TP, FP, FN, TN, Sn, Sp, Acc, Mcc, Precision, F1_score, aucs,auprc
def acc(y_test,y_predict1):
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
    return scale*tf.where(x>0.0,x,alpha*tf.nn.elu(x))


import numpy as np
import os

file_path = r"E:\GSR-ST\GSR-ST\data\all bovine_测试集_last_hidden_states.npy"
print(f"文件路径: {file_path}")
print(f"文件是否存在: {os.path.exists(file_path)}")

X_data = np.load(file_path)


w_num = X_data.shape[0]
w_len = X_data.shape[1]
w_dim = X_data.shape[2]

input_shape1 = (w_len, w_dim)
num = int(w_num / 2)
negy = np.zeros(num)
posy = np.ones(num)
y_data = np.concatenate((negy, posy), axis=0)

path=r'E:\GSR-ST\GSR-ST\ENAC\all bovine hAATAAA\test-ENAC.npy'
enac=np.load(path, allow_pickle=True)
enac=np.reshape(enac, ([enac.shape[0], 602, 4]))
enac = enac.astype(float)

path4 = r'E:\GSR-ST\GSR-ST\NCPD\all bovine hAATAAA\test-NCPD.npy'
NCPD=np.load(path4,allow_pickle=True)

NCPD_enac=np.concatenate((NCPD,enac),axis=1)
NCPD_enac=NCPD_enac.astype(float)
input_shape2 =(NCPD_enac.shape[1],NCPD_enac.shape[2])


from keras.models import  load_model
from keras_bert import  get_custom_objects
import joblib

from tensorflow.keras import activations

model1 = load_model(r'E:\GSR-ST\GSR-ST\final_final\model\all bovine hAATAAA\bert 卷积64 LSTM relu激活0.8730908081088586.h5')

model2 = load_model(r'E:\GSR-ST\GSR-ST\final_final\model\all bovine hAATAAA\NCPD_enac 卷积LSTM selu变relu 0.8511524576506526.h5')

bert_final_model = Model(inputs=model1.input, outputs=model1.get_layer('before_dense').output)
NCPD_enac_final_model = Model(inputs=model2.input, outputs=model2.get_layer('before_dense').output)

bert_final_vec = bert_final_model.predict(X_data)
NCPD_enac_final_vec =NCPD_enac_final_model.predict(NCPD_enac)
bert_NCPDenac=concatenate((bert_final_vec,NCPD_enac_final_vec))

classification_model = load_model(r"E:\GSR-ST\GSR-ST\final_model\model\all bovine hAATAAA\oe 0.8512和bert 0.8731.h5")

pred2 = classification_model.predict(bert_NCPDenac)

y_val = y_data
Twoclassfy_evalu(y_val, pred2[:,1])