import pandas as pd
import numpy as np
import os
from numpy import argmax
from numpy import array
import keras
from tensorflow.keras.utils import to_categorical
from Bio import SeqIO

from tensorflow.keras.utils import to_categorical
from collections import Counter



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

    matrix_data = np.array([list(e) for e in seq],dtype=object)  # 列出每个序列中的核苷酸
    # print(matrix_data)
    print(len(matrix_data))
    #5652
    return matrix_data

def ENAC(sequences):
    """
        计算序列的 ENAC（Expected Occurrence of Nucleotide Assignments within a Code）特征。

        参数：
        sequences (list): 包含序列的列表

        返回：
        enac (list): 序列的 ENAC 特征
        """
    AA = 'ACGT'
    AA = 'ACGT'
    enac_feature = []
    window = 5# 窗口大小
    sequences = [x.strip() for x in sequences if x.strip() != '']
    l = len(sequences)
    if l !=606:
        print(l)
        print(sequences)
    enac= []
    for i in range(0, l):
        if i < l and i + window <= l:
            count = Counter(sequences[i:i + window])
            for key in count:
                count[key] = count[key] / len(sequences[i:i + window])
            for aa in AA:
                enac.append(count[aa])

    return enac

def feature_extraction(matrix_data):
    final_feature_matrix = [ENAC(e) for e in matrix_data]
    return final_feature_matrix


matrix_data = read_fasta_file()
final_feature_matrix = feature_extraction(matrix_data)

#print(final_feature_matrix[1])
enac=np.array(final_feature_matrix,dtype=object)
save_path = '../ENAC/hAATAAA/'
os.makedirs(save_path, exist_ok=True)
np.save(save_path + 'train-ENAC.npy', enac)
