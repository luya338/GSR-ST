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
    print(TP,FN,TN,FP)
    Mcc = (TP * TN - FP * FN) / (math.sqrt(TP + FP) * math.sqrt(TP + FN) * math.sqrt(TN + FP) * math.sqrt(TN + FN))
    Precision = TP / (TP + FP)
    F1_score = (2 * Precision * Sn) / (Precision + Sn)
    fpr,tpr,thresholds = roc_curve(y_test,y_predict1)
    roc_auc=auc(fpr,tpr)
    precision, recall, thresholds = precision_recall_curve(y_test, y_predict1)
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
    print('AUPRC', auprc)
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
from tensorflow.keras.models import load_model, Model
from sklearn.model_selection import StratifiedKFold

file_path = r"D:\我的代码\启动子\train_data\S.Typhimurium\total_last_hidden_states.npy"
print(f"文件路径: {file_path}")
print(f"文件是否存在: {os.path.exists(file_path)}")
X_data = np.load(file_path)
print(X_data.shape)

w_num = X_data.shape[0]
w_len = X_data.shape[1]
w_dim = X_data.shape[2]

input_shape1 = (w_len, w_dim)
num = int(w_num / 2)
negy = np.zeros(num)
posy = np.ones(num)
y_data = np.concatenate((negy, posy), axis=0)


path = r'D:\我的代码\启动子\train_data\S.Typhimurium\ENAC.npy'
enac = np.load(path, allow_pickle=True)
enac = np.reshape(enac, ([enac.shape[0], 77, 4]))


path4 = r'D:\我的代码\启动子\train_data\S.Typhimurium\NCPD.npy'
NCPD = np.load(path4, allow_pickle=True)

NCPD_enac = np.concatenate((NCPD, enac), axis=1)
NCPD_enac = NCPD_enac.astype(float)
input_shape2 = (NCPD_enac.shape[1], NCPD_enac.shape[2])

model1 = load_model(
    r'D:\我的代码\启动子\S.Typhimurium\bert 卷积64 LSTM relu激活 fold_6_0.9061.h5')
model2 = load_model(
    r'D:\我的代码\启动子\S.Typhimurium\OE 卷积64 LSTM relu激活 fold_9_0.9483.h5')

bert_final_model = Model(inputs=model1.input, outputs=model1.get_layer('before_dense').output)
NCPD_enac_final_model = Model(inputs=model2.input, outputs=model2.get_layer('before_dense').output)

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

accuracies = []
metrics_results = {
    'Sn': [],
    'Sp': [],
    'Acc': [],
    'Mcc': [],
    'Precision': [],
    'F1_score': [],
    'AUC': [],
    'AUPRC': []
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_data, y_data)):
    print(f"开始第 {fold + 1} 折交叉验证")

    X_train, X_val = X_data[train_idx], X_data[val_idx]
    y_train, y_val = y_data[train_idx], y_data[val_idx]

    NCPD_enac_train, NCPD_enac_val = NCPD_enac[train_idx], NCPD_enac[val_idx]


    before_bert =  bert_final_model.predict(X_train)
    predict_bert =bert_final_model.predict(X_val)

    before_NCPD_enac = NCPD_enac_final_model.predict(NCPD_enac_train)
    predict_NCPD_enac = NCPD_enac_final_model.predict(NCPD_enac_val)

    bert_NCPD_enac = np.concatenate((before_bert, before_NCPD_enac), axis=1)
    bert_NCPD_enac_val = np.concatenate((predict_bert, predict_NCPD_enac), axis=1)

    combined_input = Input(shape=(768,))

    dense1 = Dense(256, activation='relu')(combined_input)
    dropout1 = Dropout(0.5)(dense1)
    output = Dense(2, activation='softmax')(dropout1)

    new_model = Model(inputs=combined_input, outputs=output)

    new_model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.0003), loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    new_model.fit(bert_NCPD_enac, y_train, epochs=1, shuffle=True,
                  validation_data=(bert_NCPD_enac_val, y_val))

    pred2 = new_model.predict(bert_NCPD_enac_val)
    accur = acc(y_val, pred2[:, 1])

    model_save_dir = './S.Typhimurium/'
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    model_path = f'{model_save_dir}/prediction relu激活 fold_{fold}_{accur:.4f}.h5'
    new_model.save(model_path)
    print(f'Model for fold {fold} saved at {model_path}')


    TP, FP, FN, TN, Sn, Sp, Acc, Mcc, Precision, F1_score, aucs, auprc = Twoclassfy_evalu(y_val, pred2[:, 1])
    metrics_results['Sn'].append(Sn)
    metrics_results['Sp'].append(Sp)
    metrics_results['Acc'].append(Acc)
    metrics_results['Mcc'].append(Mcc)
    metrics_results['Precision'].append(Precision)
    metrics_results['F1_score'].append(F1_score)
    metrics_results['AUC'].append(aucs[0])
    metrics_results['AUPRC'].append(auprc)
average_metrics = {key: np.mean(value) for key, value in metrics_results.items()}

print("十折交叉验证的平均指标：")
for metric, value in average_metrics.items():
    print(f"{metric}: {value}")