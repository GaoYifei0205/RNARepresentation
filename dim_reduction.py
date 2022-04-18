import torch
from sklearn.decomposition import PCA
from tqdm import tqdm
import os
import numpy as np

RBP_list = ['ALKBH5_Baltz2012', 'C17ORF85_Baltz2012', 'C22ORF28_Baltz2012', 'CAPRIN1_Baltz2012',
            'CLIPSEQ_AGO2', 'CLIPSEQ_ELAVL1', 'CLIPSEQ_SFRS1',
            'ICLIP_HNRNPC', 'ICLIP_TDP43', 'ICLIP_TIA1', 'ICLIP_TIAL1', 'PARCLIP_AGO1234',
            'PARCLIP_ELAVL1', 'PARCLIP_ELAVL1A', 'PARCLIP_EWSR1', 'PARCLIP_FUS',
            'PARCLIP_HUR', 'PARCLIP_IGF2BP123', 'PARCLIP_MOV10_Sievers', 'PARCLIP_PUM2',
            'PARCLIP_QKI', 'PARCLIP_TAF15', 'PTBv1', 'ZC3H7B_Baltz2012']
#4, 7


list1 = ["ls", "train"]
list2 = ["negatives", "positives"]
base_dir = "/data/gaoyifei/data/GraphProt_CLIP_sequences/"

for rbp in RBP_list:
    for l1 in list1:
        for l2 in list2:
            if not os.path.exists(os.path.join(base_dir, rbp, l1, l2, 'tensor.pt')):
                print(rbp, " ", l1, " ", l2, "Cannot find tensor.pt!")
            elif not os.path.exists(os.path.join(base_dir, rbp, l1, l2, 'reduced_tensor.pt')):
                datapath = os.path.join(base_dir, rbp, l1, l2, 'tensor.pt')
                new_path = os.path.join(base_dir, rbp, l1, l2, 'reduced_tensor.pt')
                tensor_list = torch.load(datapath)
                pca = PCA(n_components=25)
                new_tensor_list = []

                for i in tqdm(range(len(tensor_list))):
                    x = pca.fit_transform(tensor_list[0].reshape(-1, 768))
                    new_tensor_list.append(torch.tensor(x))
                    del tensor_list[0]
                torch.save(new_tensor_list, new_path)
                print(rbp, " ", l1, " ", l2, "finished!")
            else:
                print(rbp, " ", l1, " ", l2, "reduced_tensor.pt already exists!")
