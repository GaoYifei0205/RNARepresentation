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
import scipy.sparse as sp
import forgi.graph.bulge_graph as fgb
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
# tensor_list = torch.load("/data/gaoyifei/data/GraphProt_CLIP_sequences/ALKBH5_Baltz2012/ls/negatives/reduced_tensor.pt")
# print(np.shape(tensor_list[0]))
# print(np.mean(var_list))
# reshaped_list = [i.reshape(-1, 768) for i in tensor_list]
# # print(np.shape(reshaped_list[0][0]))
# print(reshaped_list[0][0])
# pca = PCA(n_components=2)
# pca.fit(X)
# print(pca.explained_variance_ratio_)

# def adj_mat(struct):
#     # create sparse matrix
#     row_col, data = [], []
#     length = len(struct)
#     for i in range(length):
#         if i != length - 1:
#             row_col.append((i, i + 1))
#             data.append(1)
#         if i != 0:
#             row_col.append((i, i - 1))
#             data.append(2)
#     bg = fgb.BulgeGraph.from_dotbracket(struct)
#     for i, ele in enumerate(struct):
#         if ele == '(':
#             row_col.append((i, bg.pairing_partner(i + 1) - 1))
#             data.append(3)
#         elif ele == ')':
#             row_col.append((i, bg.pairing_partner(i + 1) - 1))
#             data.append(4)
#     return sp.csr_matrix((data, (np.array(row_col)[:, 0], np.array(row_col)[:, 1])),
#                          shape=(length, length))
#
# struct = '.().'
# print(adj_mat(struct))
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# X, y = make_classification(n_samples=100, random_state=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
#                                                     random_state=1)
# clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
# clf.predict_proba(X_test[:1])
#
# clf.predict(X_test[:5, :])
#
# clf.score(X_test, y_test)
tensor_list = torch.load("/data/gaoyifei/data/GraphProt_CLIP_sequences/ALKBH5_Baltz2012/ls/negatives/tensor.pt")
alist = []
ALK_ls_neg = []
for i in range(len(tensor_list)):
    for j in range(len(tensor_list[i])):
        alist.append(tensor_list[i][0][j].numpy())
    ALK_ls_neg.append(np.mean(alist, axis = 0))


tensor_list = torch.load("/data/gaoyifei/data/GraphProt_CLIP_sequences/ALKBH5_Baltz2012/ls/positives/tensor.pt")
alist = []
ALK_ls_pos = []
for i in range(len(tensor_list)):
    for j in range(len(tensor_list[i])):
        alist.append(tensor_list[i][0][j].numpy())
    ALK_ls_pos.append(np.mean(alist, axis = 0))


tensor_list = torch.load("/data/gaoyifei/data/GraphProt_CLIP_sequences/ALKBH5_Baltz2012/train/negatives/tensor.pt")
alist = []
ALK_train_neg = []
for i in range(len(tensor_list)):
    for j in range(len(tensor_list[i])):
        alist.append(tensor_list[i][0][j].numpy())
    ALK_train_neg.append(np.mean(alist, axis = 0))


tensor_list = torch.load("/data/gaoyifei/data/GraphProt_CLIP_sequences/ALKBH5_Baltz2012/train/positives/tensor.pt")
alist = []
ALK_train_pos = []
for i in range(len(tensor_list)):
    for j in range(len(tensor_list[i])):
        alist.append(tensor_list[i][0][j].numpy())
    ALK_train_pos.append(np.mean(alist, axis = 0))


X_train = np.array(ALK_train_pos + ALK_train_neg)
X_test = np.array(ALK_ls_pos + ALK_ls_neg)

y_train = np.array([1 for i in range(len(ALK_train_pos))] + [0 for i in range(len(ALK_train_neg))])
y_test = np.array([1 for i in range(len(ALK_ls_pos))] + [0 for i in range(len(ALK_ls_neg))])

mlp_clf__tuned_parameters = {"hidden_layer_sizes": [(30, ),(50,), (100,)],
                                 "max_iter": [20, 30, 50, 100],
                                 "verbose": [True]
                                 }
clf = MLPClassifier()
estimator = GridSearchCV(clf, mlp_clf__tuned_parameters, n_jobs=6)
estimator.fit(X_train, y_train)

print(estimator.get_params().keys())
print(estimator.best_params_)
print(estimator.best_score_)

# clf.predict_proba(X_test[:1])
#
# clf.predict(X_test[:5, :])


