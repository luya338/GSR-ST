import tensorflow as tf
import numpy as np
import os
import pandas as pd
from transformers import BertTokenizer, BertModel
from keras.models import load_model
from keras import backend as K
import math
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Convolution1D, Softmax, Add, Lambda, MaxPooling1D, Dense, Dropout, Permute, multiply, Input, concatenate, BatchNormalization, Activation, Flatten, Bidirectional, LSTM, GRU
from tensorflow.keras import regularizers, optimizers

import torch
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


model_path = r'E:\GSR-ST\GSR-ST\6-new-12w-0\6-new-12w-0\6-new-12w-0'
if os.path.exists(model_path):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path)
else:
    raise FileNotFoundError(f"Model path not found: {model_path}")
import numpy as np
import os

file_path = r'E:\GSR-ST\GSR-ST\hg38_AATAAAtest.tsv'

data = pd.read_csv(file_path, sep='\t', header=None)
kmer_sequences = data[0].tolist()

batch_size = 128
num_batches = len(kmer_sequences) // batch_size + (1 if len(kmer_sequences) % batch_size != 0 else 0)

#all_last_hidden_states = []
#all_cls_representations = []
all_predictions = []

# 加载分类模型
classification_model = load_model(r'E:\GSR-ST\GSR-ST\final_model\model\hAATAAA\oe 0.8655和bert 0.8814.h5')

with open('E:\GSR-ST\GSR-ST\data\hg38_differ_hg19_label.txt', 'r') as file:
    labels = np.array([int(line.strip()) for line in file])
num_neg = np.sum(labels == 0)
num_pos = np.sum(labels == 1)
y_data = np.concatenate((np.zeros(num_neg), np.ones(num_pos)), axis=0)
print(f"标签总数: {len(y_data)}, 负类: {num_neg}, 正类: {num_pos}")

path=r'E:\GSR-ST\GSR-ST\ENAC\hAATAAA\hg38_AATAAA-ENAC.npy'
enac=np.load(path, allow_pickle=True)
enac=np.reshape(enac, ([enac.shape[0], 602, 4]))
print("Shape of enac:", enac.shape)
enac = enac.astype(float)

path4 = r'E:\GSR-ST\GSR-ST\NCPD\hAATAAA\hg38_AATAAA-NCPD.npy'
NCPD=np.load(path4,allow_pickle=True)
print("Shape of NCPD:", NCPD.shape)

NCPD_enac=np.concatenate((NCPD,enac),axis=1)
NCPD_enac=NCPD_enac.astype(float)
input_shape2 =(NCPD_enac.shape[1],NCPD_enac.shape[2])
input_shape1 = (512, 768)
from keras.models import  load_model
from keras_bert import  get_custom_objects
import joblib
@tf.keras.utils.register_keras_serializable()
def _gate(x):
    return x
get_custom_objects().update({'_gate': _gate})

model1 = load_model(r'E:\GSR-ST\GSR-ST\hAATAAA\bert 卷积64 LSTM0.8814159292035398.h5', custom_objects=get_custom_objects())
model2 = load_model(r'E:\GSR-ST\GSR-ST\hAATAAA\model NCPD_enacwithBiLSTM0.8654867256637168.h5',custom_objects=get_custom_objects())

bert_final_model = Model(inputs=model1.input, outputs=model1.get_layer('before_dense').output)
NCPD_enac_final_model = Model(inputs=model2.input, outputs=model2.get_layer('before_dense').output)

for i in range(num_batches):
    batch_sequences = kmer_sequences[i * batch_size: (i + 1) * batch_size]
    inputs = tokenizer.batch_encode_plus(batch_sequences, padding=True, truncation=True, max_length=512,
                                         return_tensors='pt')
    inputs = {key: val.to(model.device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_states = outputs[0].cpu().numpy()
    #print(last_hidden_states.shape)
    #all_last_hidden_states.append(last_hidden_states)
    bert_final_vec = bert_final_model.predict(last_hidden_states, batch_size=batch_size, verbose=1)

    NCPD_enac_batch = NCPD_enac[i * batch_size: (i + 1) * batch_size]
    NCPD_enac_final_vec = NCPD_enac_final_model.predict(NCPD_enac_batch, batch_size=batch_size, verbose=1)

    bert_NCPDenac = np.concatenate((bert_final_vec, NCPD_enac_final_vec), axis=-1)
    pred2 = classification_model.predict(bert_NCPDenac, batch_size=batch_size)
    all_predictions.append(pred2)

#all_last_hidden_states = np.concatenate(all_last_hidden_states, axis=0)
all_predictions = np.concatenate(all_predictions, axis=0)
np.save('all_predictions.npy', all_predictions)
print(all_predictions.shape)
y_val = y_data
Twoclassfy_evalu(y_val, all_predictions[:, 1])

