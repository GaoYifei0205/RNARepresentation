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
# seq = 'uuugcagcaguaacgaugcccuucccgagcgugcuggaguggcuccucgcugacgcaagcucaccuugcagccuccucagcaacugcaacuuaaacgcgcccucaggaagcccuggaaucucgcucagaauuuuuuuucuuuuuuugagaCAGUCUCGCUGCGACGCCCAGGCUAUAGCGCAAUGGCGCGAUCUCGGCUcccugcaaccucccucaggaagccguggaagcucgaccgccagaaacuuccuucuccugacucaggccacagucuucuguguggugagcgggguguccagcggucuccugguucccagacucggaauugggugguuugagcaaacucugg'
# print(search(seq))
import torch

tensor_list = torch.load("/data/gaoyifei/data/GraphProt_CLIP_sequences/ALKBH5_Baltz2012/ls/negatives/tensor.pt")
tensorline1 = tensor_list[0]
tensor1 = tensorline1[0]

