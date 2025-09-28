import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
from tensorflow.keras.layers import Bidirectional, GRU
#(16950, 604，50)
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Concatenate

def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.where(x > 0.0, x, alpha * tf.nn.elu(x))

def GSRwithbert(shape1=None,  filters=64, ks1=3, ks2=5, ks3=7, filter2=23, size2=6, filter3=28, size3=8,
                               filter4=11, size4=5, lstm1=189, penalty=0.005, TRAINX=None, TRAIBY=None, validX=None,
                               validY=None, lr=None):
    bert_input = Input(shape=shape1,name='bert_input')
    feature1_1 = Convolution1D(filters, ks1, padding='same', kernel_initializer='lecun_normal', kernel_regularizer=regularizers.l2(penalty))(bert_input)
    feature1_1 = BatchNormalization()(feature1_1)
    feature1_1 = Activation('selu')(feature1_1)

    feature1_2 = Convolution1D(filters, ks1+6, padding='same', kernel_initializer='lecun_normal', kernel_regularizer=regularizers.l2(penalty))(bert_input)
    feature1_2 = BatchNormalization()(feature1_2)
    feature1_2 = Activation('selu')(feature1_2)

    feature1_3 = Convolution1D(filters, ks1+12, padding='same', kernel_initializer='lecun_normal', kernel_regularizer=regularizers.l2(penalty))(bert_input)
    feature1_3 = BatchNormalization()(feature1_3)
    feature1_3 = Activation('selu')(feature1_3)

    feature = Concatenate(axis=-1)([feature1_1, feature1_2, feature1_3])
    #print(feature.shape)
    feature = MaxPooling1D(pool_size=2)(feature)
    feature = Dropout(0.42)(feature)
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
    return model