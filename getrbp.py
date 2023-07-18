from tqdm import tqdm


result=["ALKBH5_Baltz2012",
"C17ORF85_Baltz2012",
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


datapath = '/home/gaoyifei/PrismNet/data/uniprot_sprot.fasta'

#New machine
result1 = ['ELAVL1']
for name in tqdm(result1):

    with open(datapath, 'r') as f, open('/amax/data/gaoyifei/GraphProt/GraphProt_CLIP_sequences/PTBv1/PTBv1.fa', 'w') as out_f:
        # 初始化一个空字符串变量，用于存储当前读取的记录
        current_record = ''
        # 初始化一个变量，用于记录当前记录的名字
        current_name = ''
        # new_name = name[:name.find('_')]
        new_name = name

        # 遍历文件中的每一行
        for line in tqdm(f):
            # 如果当前行是一个新记录的开始
            if line.startswith('>'):
                current_name = line.strip()
                if len(current_record)>0:
                    print(current_record)
                    out_f.write(f'{current_record}\n')
                    current_record = ''
            if 'PTB' in current_name:
            # if ('GN='+new_name) in current_name and 'HUMAN' in current_name:
                if line.startswith('>'):
                    print(current_name)
                    out_f.write(f'{current_name}\n')
                else:
                    current_record += line.strip()

# result1 = ['U2AF65_Hela']
# for name in tqdm(result1):
#     with open(datapath, 'r') as f, open('/data/gaoyifei/data/PrismNetData/'+name+'/'+name+'.fa', 'w') as out_f:
#         # 初始化一个空字符串变量，用于存储当前读取的记录
#         current_record = ''
#         # 初始化一个变量，用于记录当前记录的名字
#         current_name = ''
#         new_name = name[:name.find('_')]

#         # 遍历文件中的每一行
#         for line in tqdm(f):
#             # 如果当前行是一个新记录的开始
#             if line.startswith('>'):
#                 current_name = line.strip()
#                 if len(current_record)>0:
#                     print(current_record)
#                     out_f.write(f'{current_record}\n')
#                     current_record = ''
#             if 'U2AF2_HUMAN' in current_name:
#             # if ('GN='+new_name) in current_name and 'HUMAN' in current_name:
#                 if line.startswith('>'):
#                     print(current_name)
#                     out_f.write(f'{current_name}\n')
#                 else:
#                     current_record += line.strip()





            # else:
            #     current_record += line.strip()
            # 如果当前记录同时包含 name 和 "HUMAN"，则将其保存到输出文件中

            # if 'AGO2_HUMAN' in current_name:
            # # if new_name in current_name and 'HUMAN' in current_name:
            #     print(current_name)
            #     out_f.write(f'{current_name}{current_record}\n')

        # # 处理最后一个记录
        # if new_name in current_name and 'HUMAN' in current_name:
        #     out_f.write(f'{current_name}\n{current_record}\n')

