# def search(seq: str):
#     d = {}
#     with open("/data/gaoyifei/data/GraphProt_CLIP_sequences/ALKBH5_Baltz2012/ls/negatives/structure.txt") as file:
#         key = None
#         for line in file:
#             if line.startswith(">"):
#                 continue
#             if key:
#                 l = line.split(" ")
#                 d[key] = l[0]
#                 key = None
#             else:
#                 key = line[:-1]
#     return d[seq]

# seq = 'ttatcagtattgtttcatttttaaatgatttttgaagcttaacttattacttagtgtctaaaacatagttaatgtcctccaaaatgtcatatgaataatgtacatagagatgtatgtctaaagttaaaaatgttacagtgtactagtaacATCTTGTTTTGGTCAGAAAAGTGTGTAGagtacttgtatttttttcattcaaaatagaaaatttggtgttattatagaaattgatttaaaatgtttttatttgactgatttatgaatgagaaatgaacttctttgatgtttcaggatttttacttcattttttttcctcagtaaaaaa'
# l = seq.replace('T', 'U')
# l = l.replace('t', 'u')
# print(l)

import torch
from sklearn.decomposition import PCA
import numpy as np
# tensor_list = torch.load("/data/gaoyifei/data/GraphProt_CLIP_sequences/ALKBH5_Baltz2012/ls/negatives/tensor.pt")
# pca = PCA(n_components=25)
# new_tensor_list = []
# # var_list = []
# for i in tensor_list:
#     x = pca.fit_transform(i.reshape(-1, 768))
#
#     new_tensor_list.append(torch.tensor(x))
#     torch.save(new_tensor_list, "/data/gaoyifei/data/GraphProt_CLIP_sequences/ALKBH5_Baltz2012/ls/negatives/reduced_tensor.pt")
#     # print(sum(pca.explained_variance_ratio_))
#     # var_list.append(sum(pca.explained_variance_ratio_))
# print(np.shape(new_tensor_list[0]))
tensor_list = torch.load("/data/gaoyifei/data/GraphProt_CLIP_sequences/ALKBH5_Baltz2012/ls/negatives/tensor.pt")
print(np.shape(tensor_list[0]))
print(np.shape(tensor_list[0][0]))
# print(np.mean(var_list))
# reshaped_list = [i.reshape(-1, 768) for i in tensor_list]
# # print(np.shape(reshaped_list[0][0]))
# print(reshaped_list[0][0])
# pca = PCA(n_components=2)
# pca.fit(X)
# print(pca.explained_variance_ratio_)



