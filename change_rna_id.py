import re

# 读取文件内容
with open('/amax/data/gaoyifei/GraphProt/GraphProt_CLIP_sequences/PTBv1/train/positives/data.fa', 'r') as file:
    data = file.read()

# 使用正则表达式替换每一条数据
pattern = r'>[^;]+;'
replacement = '>PTBv1.train.pos;'
result = re.sub(pattern, replacement, data)

# 将替换后的数据写入新文件
with open('/amax/data/gaoyifei/GraphProt/GraphProt_CLIP_sequences/PTBv1/train/positives/output_file.fa', 'w') as output_file:
    output_file.write(result)