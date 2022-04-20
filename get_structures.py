import subprocess
import os
# RBP_list = ['ALKBH5_Baltz2012', 'C17ORF85_Baltz2012', 'C22ORF28_Baltz2012', 'CAPRIN1_Baltz2012',
#             'CLIPSEQ_AGO2', 'CLIPSEQ_ELAVL1', 'CLIPSEQ_SFRS1',
#             'ICLIP_HNRNPC', 'ICLIP_TDP43', 'ICLIP_TIA1', 'ICLIP_TIAL1', 'PARCLIP_AGO1234',
#             'PARCLIP_ELAVL1', 'PARCLIP_ELAVL1A', 'PARCLIP_EWSR1', 'PARCLIP_FUS',
#             'PARCLIP_HUR', 'PARCLIP_IGF2BP123', 'PARCLIP_MOV10_Sievers', 'PARCLIP_PUM2',
#             'PARCLIP_QKI', 'PARCLIP_TAF15', 'PTBv1', 'ZC3H7B_Baltz2012']

RBP_list = ['ALKBH5_Baltz2012', 'C17ORF85_Baltz2012', 'C22ORF28_Baltz2012', 'CAPRIN1_Baltz2012',
            'CLIPSEQ_ELAVL1', 'CLIPSEQ_SFRS1', 'ICLIP_HNRNPC', 'ICLIP_TIA1',
            'PARCLIP_ELAVL1', 'PARCLIP_ELAVL1A', 'PARCLIP_EWSR1',
            'PARCLIP_IGF2BP123', 'PARCLIP_MOV10_Sievers', 'PARCLIP_PUM2',
            'PARCLIP_QKI', 'PARCLIP_TAF15', 'ZC3H7B_Baltz2012']

list1 = ["ls", "train"]
list2 = ["negatives", "positives"]
base_dir = "/data/gaoyifei/data/GraphProt_CLIP_sequences/"
for l in RBP_list:
    for l1 in list1:
        for l2 in list2:
            path = base_dir + l + '/' + l1 + '/' + l2
            with open(os.path.join(path, 'structure.txt')) as file:
                if len(file.readlines()) == 0:
                    subprocess.call(
                        f"/home/gaoyifei/.conda/envs/mxfold/bin/mxfold2 predict {path}/data.fa > {path}/structure.txt",
                        shell=True)
                    print(l, " ", l1, " ", l2, "finished!")
                else:
                    print(l, " ", l1, " ", l2, "already exists!")

