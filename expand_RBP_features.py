import torch
import os
# RBP = 'C17ORF85_Baltz2012'
# given_feature = torch.load('/amax/data/gaoyifei/GraphProt/GraphProt_CLIP_sequences/'+ RBP+'/'+RBP+'.pt')['representations'][12]
# target_shape = (501, 768)
# m, _ = given_feature.shape
# print(given_feature.shape)
# if m > target_shape[0]:
#     final_feature = given_feature[:target_shape[0], :]

# # 如果m小于501，则在后面补零
# elif m < target_shape[0]:
#     num_zero_rows = target_shape[0] - m
#     zero_rows = torch.zeros((num_zero_rows, 768))
#     final_feature = torch.cat((given_feature, zero_rows), dim=0)
# print(final_feature.shape)
# print(final_feature)

# expand_path = '/amax/data/gaoyifei/GraphProt/GraphProt_CLIP_sequences/'+ RBP+'/'+RBP+'_expand.pt'
# torch.save(final_feature, expand_path)

result=[
# "ALKBH5_Baltz2012",
# "C17ORF85_Baltz2012",
"C22ORF28_Baltz2012",
"CAPRIN1_Baltz2012",
"CLIPSEQ_AGO2",
"CLIPSEQ_ELAVL1",
"CLIPSEQ_SFRS1",
"ICLIP_HNRNPC",
"ICLIP_TDP43",
"ICLIP_TIA1",
"ICLIP_TIAL1",
# "PARCLIP_AGO1234",
"PARCLIP_ELAVL1",
"PARCLIP_ELAVL1A",
"PARCLIP_EWSR1",
"PARCLIP_FUS",
"PARCLIP_HUR",
# "PARCLIP_IGF2BP123",
"PARCLIP_MOV10_Sievers",
"PARCLIP_PUM2",
"PARCLIP_QKI",
"PARCLIP_TAF15",
"PTBv1",
"ZC3H7B_Baltz2012"]

base_dir = '/amax/data/gaoyifei/GraphProt/GraphProt_CLIP_sequences/'
for RBP in result:
    print(RBP)
    given_feature = torch.load(base_dir+ RBP+'/'+RBP+'.pt')['representations'][12]
    target_shape = (501, 768)
    m, _ = given_feature.shape
    print(given_feature.shape)
    if m > target_shape[0]:
        final_feature = given_feature[:target_shape[0], :]

    # 如果m小于501，则在后面补零
    elif m < target_shape[0]:
        num_zero_rows = target_shape[0] - m
        zero_rows = torch.zeros((num_zero_rows, 768))
        final_feature = torch.cat((given_feature, zero_rows), dim=0)
    print(final_feature.shape)
    # print(final_feature)

    expand_path = '/amax/data/gaoyifei/GraphProt/GraphProt_CLIP_sequences/'+ RBP+'/'+RBP+'_expand.pt'
    torch.save(final_feature, expand_path)