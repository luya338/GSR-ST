import pandas as pd
import numpy as np
import os

from numpy import argmax
from numpy import array
import keras
from tensorflow.keras.utils import to_categorical
from Bio import SeqIO


def read_fasta_file():
    fh = open(r'E:\GSR-ST\GSR-ST\processed_data\PAS_data\train\total\hAATAAA.fa', 'r')
    seq = []
    for line in fh:
        if line.startswith('>'):
            continue
        else:
            if len(line)>2:
                seq.append(line.replace('\n', '').replace('\r', ''))  # \r\n 一般一起用，用来表示键盘上的回车键，也可只用 \n。
            else:
                continue
    fh.close()

    matrix_data = np.array([list(e) for e in seq],dtype=object)
    return matrix_data


def extract_line(data_line):
    """
        将 DNA 序列转换为特征向量表示。

        参数:
            data_line (str): DNA 序列的字符串形式。

        返回:
            one_line_feature (list): 转换后的特征向量列表。
        """
    A = [0, 0, 0, 1]
    T = [0, 0, 1, 0]
    C = [0, 1, 0, 0]
    G = [1, 0, 0, 0]
    feature_representation = {"A": A, "C": C, "G": G, "T": T}
    one_line_feature = []
    # 枚举序列中的每个核苷酸，获取其索引和值
    for index, data in enumerate(data_line):
        # print(index, data)
        if data in feature_representation.keys():
            one_line_feature.extend(feature_representation[data])
    return one_line_feature


def feature_extraction(matrix_data):
    final_feature_matrix = [extract_line(e) for e in matrix_data]
    return final_feature_matrix

matrix_data = read_fasta_file()
final_feature_matrix = feature_extraction(matrix_data)
#将特征向量保存为numpy数组
Onehot = np.array(final_feature_matrix, dtype=object)
save_path = '../one-hot/hAATAAA/'
os.makedirs(save_path, exist_ok=True)
np.save(save_path + 'train-one-hot.npy', Onehot)
