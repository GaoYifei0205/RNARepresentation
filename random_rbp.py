# get random rbp data from dataset

import random

base_dir = '/amax/data/gaoyifei/GraphProt/GraphProt_CLIP_sequences/'

trains = ["ALKBH5_Baltz2012",
# "C17ORF85_Baltz2012",
"C22ORF28_Baltz2012",
"CAPRIN1_Baltz2012",
"CLIPSEQ_AGO2",
"CLIPSEQ_ELAVL1",
"CLIPSEQ_SFRS1",
"ICLIP_HNRNPC",
# "ICLIP_TDP43",
"ICLIP_TIA1",
"ICLIP_TIAL1",
"PARCLIP_AGO1234",
"PARCLIP_ELAVL1",
"PARCLIP_ELAVL1A",
# "PARCLIP_EWSR1",
"PARCLIP_FUS",
"PARCLIP_HUR",
"PARCLIP_IGF2BP123",
# "PARCLIP_MOV10_Sievers",
"PARCLIP_PUM2",
"PARCLIP_QKI",
"PARCLIP_TAF15",
"PTBv1",
"ZC3H7B_Baltz2012"]



# rbp1 = base_dir+'C17ORF85_Baltz2012'
# rbp2 = base_dir+'ICLIP_TDP43'
# rbp3 = base_dir+'ICLIP_TIAL1'
# rbp4 = base_dir+'PARCLIP_MOV10_Sievers'
# 定义文件路径
file_paths_neg = [base_dir+trains[i]+'/train/negatives/data.fa' for i in range(len(trains))]
file_paths_pos = [base_dir+trains[i]+'/train/positives/data.fa' for i in range(len(trains))]

# 定义每个文件要抽取的数据数量
samples_per_file = 1000

# 初始化抽样结果列表
sampled_data_neg = []

# 遍历每个文件路径
for file_path in file_paths_neg:
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # 将每两行数据合并为一条数据
        sequences = [''.join(lines[i:i+2]) for i in range(0, len(lines), 2)]
        
        # 随机抽取指定数量的数据
        random_samples = random.sample(sequences, samples_per_file)
        sampled_data_neg.extend(random_samples)

# # 打印抽样结果
# for idx, data in enumerate(sampled_data_neg, start=1):
#     print(f"Sample {idx}:\n{data}")

output_file_path_neg = base_dir+'sampled_data_neg.fa'

# 将抽样数据存储到新的fa文件中
with open(output_file_path_neg, 'w') as output_file_neg:
    for data in sampled_data_neg:
        output_file_neg.write(data)

# 初始化抽样结果列表
sampled_data_pos = []

# 遍历每个文件路径
for file_path in file_paths_pos:
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # 将每两行数据合并为一条数据
        sequences = [''.join(lines[i:i+2]) for i in range(0, len(lines), 2)]
        
        # 随机抽取指定数量的数据
        random_samples = random.sample(sequences, samples_per_file)
        sampled_data_pos.extend(random_samples)

# 打印抽样结果
# for idx, data in enumerate(sampled_data_pos, start=1):
#     print(f"Sample {idx}:\n{data}")

output_file_path_pos = base_dir+'sampled_data_pos.fa'

# 将抽样数据存储到新的fa文件中
with open(output_file_path_pos, 'w') as output_file_pos:
    for data in sampled_data_pos:
        output_file_pos.write(data)