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
# tensor_list = torch.load("/data/gaoyifei/data/GraphProt_CLIP_sequences/ALKBH5_Baltz2012/ls/negatives/reduced_tensor.pt")
# print(np.shape(tensor_list[0]))
# print(np.mean(var_list))
# reshaped_list = [i.reshape(-1, 768) for i in tensor_list]
# # print(np.shape(reshaped_list[0][0]))
# print(reshaped_list[0][0])
# pca = PCA(n_components=2)
# pca.fit(X)
# print(pca.explained_variance_ratio_)
# string = 'attgctattaaagctttactgtggagggtggagtttcaagagtgttaagcatggtgactgtttctgtctttgccattgggtgctaagaatgattgactaaaccaagcaggaaagatttctttgctttcccaatactttgcaaatcttgttATACTAACTAGTCTGCTGTTATACtcttatcatctcttactcctctgactcagaatattctactgtatagggtgaatacttttggtatccaccctccccctccatactggaaagtactttcagggtacttagttcattttacaaatacaaaactgaggcctggattacaaaaag'
# new_string = string.upper()
# print(new_string)
# import numpy as np
#
# a = np.load('/home/gaoyifei/RNASSR-Net-main/data/GraphProt_CLIP_sequences/ALKBH5_Baltz2012/ls/negatives/ALKBH5_Baltz2012.train.neg_380.npy')
#
# from sklearn.neural_network import MLPClassifier
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
# # X, y = make_classification(n_samples=100, random_state=1)
# # X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
# #                                                     random_state=1)
# # clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
# # clf.predict_proba(X_test[:1])
# #
# # clf.predict(X_test[:5, :])
# #
# # clf.score(X_test, y_test)
tensor_list = []
tensor_list = torch.load("/data/gaoyifei/data/GraphProt_CLIP_sequences/C17ORF85_Baltz2012/ls/negatives/tensor.pt")
alist = []
C17ORF85_ls_neg = []
for i in range(len(tensor_list)):
    for j in range(len(tensor_list[i])):
        alist.append(tensor_list[i][0][j].numpy())
    C17ORF85_ls_neg.append(np.mean(alist, axis = 0))

tensor_list = []
tensor_list = torch.load("/data/gaoyifei/data/GraphProt_CLIP_sequences/C17ORF85_Baltz2012/ls/positives/tensor.pt")
alist = []
C17ORF85_ls_pos = []
for i in range(len(tensor_list)):
    for j in range(len(tensor_list[i])):
        alist.append(tensor_list[i][0][j].numpy())
    C17ORF85_ls_pos.append(np.mean(alist, axis = 0))

tensor_list = []
tensor_list = torch.load("/data/gaoyifei/data/GraphProt_CLIP_sequences/C17ORF85_Baltz2012/train/negatives/tensor.pt")
alist = []
C17ORF85_train_neg = []
for i in range(len(tensor_list)):
    for j in range(len(tensor_list[i])):
        alist.append(tensor_list[i][0][j].numpy())
    C17ORF85_train_neg.append(np.mean(alist, axis = 0))

tensor_list = []
tensor_list = torch.load("/data/gaoyifei/data/GraphProt_CLIP_sequences/C17ORF85_Baltz2012/train/positives/tensor.pt")
alist = []
C17ORF85_train_pos = []
for i in range(len(tensor_list)):
    for j in range(len(tensor_list[i])):
        alist.append(tensor_list[i][0][j].numpy())
    C17ORF85_train_pos.append(np.mean(alist, axis = 0))


X_train = np.array(C17ORF85_train_pos + C17ORF85_train_neg)
# X_train = X_train.reshape([-1, 768, 1])
X_test = np.array(C17ORF85_ls_pos + C17ORF85_ls_neg)
# X_test = X_test.reshape([-1, 768, 1])

y_train = np.array([1 for i in range(len(C17ORF85_train_pos))] + [0 for i in range(len(C17ORF85_train_neg))])
y_test = np.array([1 for i in range(len(C17ORF85_ls_pos))] + [0 for i in range(len(C17ORF85_ls_neg))])


# print(len(y_test))
# mlp_clf__tuned_parameters = {"hidden_layer_sizes": [(30, )],
#                                  "max_iter": [20]
#                                  }
# clf = MLPClassifier()
# estimator = GridSearchCV(clf, mlp_clf__tuned_parameters)
# estimator.fit(X_train, y_train)
# print(estimator.score(X_test, y_test))

import tensorflow
import numpy as np
import tensorflow.keras
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model, Sequential # basic class for specifying and training a neural network
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense, Dropout, Activation, Flatten
from tensorflow.keras.callbacks import EarlyStopping

callback = EarlyStopping(monitor='val_loss', patience=3)

batch_size = 16
num_epochs = 5    # we iterate 15 times over the entire training set
# kernel_size = 5    # we will use 5x5 kernels throughout
# pool_size = 2      # we will use 2x2 pooling throughout
# conv_depth_1 = 6   # we will initially have 6 kernels in first conv. layer...
# conv_depth_2 = 16  # ...switching to 16 after the first pooling layer
# drop_prob_1 = 0.   # dropout after pooling with probability 0.
# drop_prob_2 = 0.   # dropout in the FC layer with probability 0.
# hidden_size = 128  # the FC layer will have 128neurons
# weight_penalty = 0. # Factor for weights penalty
num_classes = 2

# model = Sequential([
#     # Conv1D(600, 3, activation='relu'),
#     # MaxPooling1D(),
#     # Conv1D(300, 5, activation='relu'),
#     # MaxPooling1D(),
#     Conv1D(10, 5, padding = 'same', activation='relu'),
#     MaxPooling1D(),
#     Flatten(),
#     # Dense(30, activation='relu'),
#     Dense(1, activation='sigmoid')
# ])
#沿着RNA方向滑动。滑动窗口得到的窗口长度还是501.要两边padding一下，确保滑动之后还是501维。
# print(model.summary())

# Loss function and Optimizer
# print(y_train)
# model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
#               optimizer='adam', # using the Adam optimiser
#               metrics=['accuracy']) # reporting the accuracy
# # Training
# history = model.fit(X_train, y_train, # Train the model using the training set...
#           batch_size=batch_size, epochs=num_epochs, callbacks=[callback],
#           verbose=1, validation_split=0.2) # ...holding out 40% of the data for validation
# # Evaluation
# for loss_name, loss_value in list(zip(model.metrics_names, model.evaluate(X_test, y_test, verbose=1))):
#     print('The final {} on the TEST set is: {:.2f}.'.format(loss_name, loss_value)) # Evaluate the trained model on the test set!


# a = np.load('/home/gaoyifei/RNASSR-Net-main/data/GraphProt_CLIP_sequences/ALKBH5_Baltz2012/ls/negatives/ALKBH5_Baltz2012.train.neg_380.npy')
#
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
print(clf.predict_proba(X_test[:1]))

print(clf.predict(X_test[:5, :]))

print(clf.score(X_test, y_test))










