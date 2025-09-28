import numpy as np
import os

import numpy as np
import os


def read_fasta_file():
    """
    从 FASTA 文件中读取序列数据。
    返回:
    seq (list): 从文件中读取的序列列表
    """
    with open(r'E:\GSR-ST\GSR-ST\processed_data\PAS_data\train\total\hAATAAA.fa', 'r') as fh:
        seq = []
        for line in fh:
            if line.startswith('>'):
                continue
            else:
                if len(line) > 2:
                    seq.append(line.replace('\n', '').replace('\r', ''))  # 去掉换行符和回车符
                else:
                    continue
    return seq


def custom_ncpd_encode_single(seqs):
    """
    为多个DNA序列进行 NCPD 编码
    返回 (L, M, 1) 的 numpy 数组，表示每个位置的编码值
    """
    encoding = []
    for seq in seqs:
        seq_length = len(seq)
        bases = ['A', 'C', 'G', 'T']
        encoding_seq = [0] * seq_length
        counters = {base: 0 for base in bases}

        for i, char in enumerate(seq):
            if char in bases:
                counters[char] += 1
                encoding_seq[i] = counters[char] / (i + 1)

        encoding.append(np.array(encoding_seq).reshape(-1, 1))

    return np.array(encoding)  # 返回 (L, M, 1) 的三维数组


def to_properties_code_NCP(seqs):
    properties_code_dict = {
        'A': [1, 1, 1], 'C': [0, 1, 0], 'G': [1, 0, 0], 'T': [0, 0, 1],
        'a': [1, 1, 1], 'c': [0, 1, 0], 'g': [1, 0, 0], 't': [0, 0, 1]
    }

    properties_code = []
    for seq in seqs:
        seq_length = len(seq)
        properties_matrix = np.zeros([seq_length, 3], dtype=float)
        m = 0
        for seq_base in seq:
            properties_matrix[m, :3] = properties_code_dict[seq_base]
            m += 1
        properties_code.append(properties_matrix)

    return np.array(properties_code)  # 返回 (L, M, N) 的三维数组


# 读取文件
seq = read_fasta_file()
# 生成 NCP 和 ND 编码
NCP = to_properties_code_NCP(seq)
ND = custom_ncpd_encode_single(seq)
#print(f"NCP shape: {NCP.shape}")
#print(f"ND shape: {ND.shape}")

final_encoding = np.concatenate((NCP, ND), axis=-1)  # 在最后一个维度上拼接

NCPD = np.array(final_encoding, dtype=object)
save_path = '../NCPD/hAATAAA/'
os.makedirs(save_path, exist_ok=True)  # 创建文件夹
np.save(save_path + 'train-NCPD.npy', NCPD)

#print(np.array(final_encoding).shape)  # 打印最终编码的形状

