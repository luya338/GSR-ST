import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.keras.layers import Bidirectional, GRU, MaxPooling1D, Dropout
from tensorflow.keras.layers import Bidirectional, LSTM
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
    if TP==0 or FN==0 or TN==0 or FP==0:
        print('TP==0 or FN==0 or TN==0 or FP==0')
        return Sn,Sp,Acc
    else:
        Mcc = (TP * TN - FP * FN) / (math.sqrt(TP + FP) * math.sqrt(TP + FN) * math.sqrt(TN + FP) * math.sqrt(TN + FN))
        Precision = TP / (TP + FP)
        F1_score = (2 * Precision * Sn) / (Precision + Sn)
        fpr,tpr,thresholds = roc_curve(y_test,y_predict1)
        roc_auc=auc(fpr,tpr)
        precision, recall, thresholds = precision_recall_curve(y_test, y_predict1)
        auprc = auc(recall, precision)
        aucs.append(roc_auc)
        #画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数计算出来
        # plt.plot(fpr,tpr,lw=1,alpha=0.3,label='ROC fold %d(area=%0.2f)'% (i,roc_auc))
        # i +=1
        # plt.plot([0,1],[0,1],linestyle='--',lw=2,color='r',label='Luck',alpha=.8)


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
        print('AUPRC', auprc)

        # result = [TP, FP, FN, TN, Sn, Sp, Acc, Mcc, Precision, F1_score, aucs ]
        # np.savetxt('F:/yuanyuan/AAA/iLearn-master/sequencefeature/CNN_Test_result.txt', result, delimiter=" ", fmt='%s')
        return  TP, FP, FN, TN, Sn, Sp, Acc, Mcc, Precision, F1_score, aucs, auprc
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
    return scale * tf.where(x > 0.0, x, alpha * tf.nn.elu(x))


import numpy as np
import os

file_path = r"E:\GSR-ST\GSR-ST\hAATAAA 训练集_last_hidden_states.npy"
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

path = r'E:\GSR-ST\GSR-ST\ENAC\hAATAAA\train-ENAC.npy'
enac = np.load(path, allow_pickle=True)
enac=np.reshape(enac, ([enac.shape[0], 602, 4]))
path4 = r'E:\GSR-ST\GSR-ST\NCPD\hAATAAA\train-NCPD.npy'
NCPD = np.load(path4, allow_pickle=True)

NCPD_enac=np.concatenate((NCPD,enac),axis=1)
NCPD_enac=NCPD_enac.astype(float)
input_shape2 =(NCPD_enac.shape[1],NCPD_enac.shape[2])

seed=7
np.random.seed(seed)

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import MultiHeadAttention

@tf.keras.utils.register_keras_serializable()
def _gate(x):
    return x

get_custom_objects().update({'_gate': _gate})


model1 = load_model(r'E:\GSR-ST\GSR-ST\final_final\model\hAATAAA\bert 卷积64 LSTM0.8814159292035398.h5')
model2 = load_model(r'E:\GSR-ST\GSR-ST\final_final\model\hAATAAA\model NCPD_enacwithBiLSTM0.8654867256637168.h5')
w_train, w_val, s_train, s_val, y_train, y_val = train_test_split(
    X_data, NCPD_enac, y_data, test_size=0.2, random_state=7
)

bert_final_model = Model(inputs=model1.input, outputs=model1.get_layer('before_dense').output)
NCPD_enac_final_model = Model(inputs=model2.input, outputs=model2.get_layer('before_dense').output)

batch_size = 1000
before_bert = []
for i in range(0, len(w_train), batch_size):
    batch = w_train[i:i+batch_size]
    predictions = bert_final_model.predict(batch)
    before_bert.append(predictions)
before_bert = np.concatenate(before_bert, axis=0)

batch_size = 1000
predict_bert = []
for i in range(0, len(w_val), batch_size):
   batch = w_val[i:i + batch_size]
   batch_prediction = bert_final_model.predict(batch)
   predict_bert.append(batch_prediction)
predict_bert = np.concatenate(predict_bert, axis=0)

print(before_bert.shape)

before_NCPD_enac  = NCPD_enac_final_model.predict(s_train)
predict_NCPD_enac  = NCPD_enac_final_model.predict(s_val)
print(before_NCPD_enac.shape)

bert_NCPD_enac=concatenate((before_bert,before_NCPD_enac),axis=1)
bert_NCPD_enac_val = concatenate((predict_bert, predict_NCPD_enac),axis=1)


combined_input = Input(shape=(768,))

dense1 = Dense(256, activation='relu', name="dense1")(combined_input)
dropout1 = Dropout(0.5)(dense1)
output = Dense(2, activation='softmax')(dropout1)

new_model = Model(inputs=combined_input, outputs=output)
new_model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.0003), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
new_model.summary()

history = new_model.fit(bert_NCPD_enac, y_train,
                        epochs=1,
                        batch_size=128,
                        shuffle=True,
                        validation_data=(bert_NCPD_enac_val, y_val))


# 评估模型
pred2 =new_model.predict(bert_NCPD_enac_val)
print(pred2.shape)
Twoclassfy_evalu(y_val, pred2[:,1])
new_model.save('./final_final/model/hAATAAA/全连接.h5')