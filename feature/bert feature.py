import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import os

model_path = r'E:\GSR-ST\GSR-ST\6-new-12w-0\6-new-12w-0\6-new-12w-0'
if os.path.exists(model_path):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path)
else:
    raise FileNotFoundError(f"Model path not found: {model_path}")

# 定义数据集文件的路径
file_path = r'E:\GSR-ST\GSR-ST\sample_data\6mer_hAATAAA_train.tsv'

data = pd.read_csv(file_path, sep='\t', header=None)
kmer_sequences = data[0].tolist()


batch_size = 128  # 根据内存情况调整此值
num_batches = len(kmer_sequences) // batch_size + (1 if len(kmer_sequences) % batch_size != 0 else 0)

# 用于存储最后的输出
all_last_hidden_states = []
all_cls_representations = []


for i in range(num_batches):
    batch_sequences = kmer_sequences[i * batch_size: (i + 1) * batch_size]
    inputs = tokenizer.batch_encode_plus(batch_sequences, padding=True, truncation=True, max_length=512,
                                         return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_states = outputs[0]
    cls_representations = last_hidden_states[:, 0, :]

    all_last_hidden_states.append(last_hidden_states.numpy())
    all_cls_representations.append(cls_representations.numpy())

all_last_hidden_states = np.concatenate(all_last_hidden_states, axis=0)
all_cls_representations = np.concatenate(all_cls_representations, axis=0)

save_dir = "E:\\data\\"  # 确保目录存在

np.save(save_dir + "hAATAAA_训练集_last_hidden_states.npy", all_last_hidden_states)
np.save(save_dir + "hAATAAA_训练集_cls_representations.npy", all_cls_representations)