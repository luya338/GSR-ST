import numpy as np
import os
import csv

def read_fasta_file():
    """
    从 FASTA 文件中读取序列数据。

    返回:
    seq (list): 从文件中读取的序列列表
    """
    # 打开指定文件
    fh = open(r'E:\GSR-ST\GSR-ST\processed_data\PAS_data\train\total\hAATAAA.fa', 'r')

    seq = []
    for line in fh:
        if line.startswith('>'):
            continue
        else:
            if len(line) > 2:
                seq.append(line.replace('\n', '').replace('\r', ''))  # \r\n 一般一起用，用来表示键盘上的回车键，也可只用 \n。
            else:
                continue
    fh.close()

    return seq


def seq2kmer(seq, k):
    """
    将原始序列转换为k-mers

    参数:
    seq -- str, 原始序列。
    k -- int, 指定的k-mer长度。

    返回:
    kmers -- str, 以空格分隔的k-mers
    """
    kmer = [seq[x:x + k] for x in range(len(seq) + 1 - k)]
    kmers = " ".join(kmer)
    return kmers


def generate_kmers_for_sequences(sequence_list, k):
    """
    对序列列表中的每个序列生成 k-mer 表示。

    参数:
    sequence_list -- list，包含多个序列的列表。
    k -- int，k-mer 的长度。

    返回:
    kmers_list -- list，包含每个序列的 k-mer 表示的列表。
    """
    kmers_list = []
    for seq in sequence_list:
        kmers = seq2kmer(seq, k)
        kmers_list.append(kmers)
    return kmers_list


def save_kmers_to_tsv(kmers_list, output_file):
    """
    将生成的 k-mer 列表保存为 TSV 文件。

    参数:
    kmers_list -- list，包含每个序列的 k-mer 表示的列表。
    output_file -- str，输出的 TSV 文件路径。

    """
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        for kmers in kmers_list:
            writer.writerow([kmers])  # 每一行写入一个 k-mer 表示


seq = read_fasta_file()
seq = [x.strip() for x in seq if x.strip() != '']
l = len(seq)
if l != 606:
    print(l)
    print(seq)

kmers_list = generate_kmers_for_sequences(seq, 6)


for idx, kmers in enumerate(kmers_list):
    print(f"序列 {idx+1} 的 k-mer 表示: {kmers}")

# 将 k-mer 表示保存为 TSV 文件
output_file = ('6mer_hAATAAA_train.tsv')
save_kmers_to_tsv(kmers_list, output_file)

print(f"k-mers 列表已保存到 {output_file}")


