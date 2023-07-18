from tqdm import tqdm
import os
import subprocess
import torch
import pickle
import shutil

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


def runcmd(command):
    ret = subprocess.run(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding="utf-8",timeout=1)
    if ret.returncode == 0:
        print("success:",ret)
    else:
        print("error:",ret)

def cmd(command):
    subp = subprocess.Popen(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding="utf-8")
    subp.wait()
    if subp.poll() == 0:
        print(subp.communicate()[1])
    else:
        print("失败")

os.environ["CUDA_VISIBLE_DEVICES"]='1'


common_cmd1 = 'python -u esm/scripts/extract.py esm1_t12_85M_UR50S '
common_cmd2 = ' --repr_layers 12 --include mean per_tok > '
common_cmd3 = ' 2>&1 &'
RBP_PATH = '/amax/data/gaoyifei/GraphProt/GraphProt_CLIP_sequences/'
# for name in tqdm(result[2:3]):
#     this_dir = os.path.join(RBP_PATH, name)
#     this_rbp = os.path.join(RBP_PATH, name, name+'.fa ')
#     log_path = os.path.join(RBP_PATH, name, 'emb_log.out')
#     cmd(common_cmd1 + this_rbp + this_dir + common_cmd2 + log_path + common_cmd3)
#     print(name, " finished!")

haventdone = []
for name in tqdm(result):
    folder_path = '/amax/data/gaoyifei/GraphProt/GraphProt_CLIP_sequences/'+name  # 文件夹路径
    # 获取文件夹内所有文件
    file_list = os.listdir(folder_path)
    flag = 0
    for filename in file_list:
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.endswith('.pt'):
            new_filename = os.path.join(folder_path, name+'.pt')
            # 重命名文件
            shutil.move(file_path, new_filename)
            print(name, "renamed.")
            flag = 1
            break
    if(flag  == 0):
        print(name, "doesn't have pt file")
        haventdone.append(name)

for name in tqdm(haventdone):
    this_dir = os.path.join(RBP_PATH, name)
    this_rbp = os.path.join(RBP_PATH, name, name+'.fa ')
    log_path = os.path.join(RBP_PATH, name, 'emb_log.out')
    cmd(common_cmd1 + this_rbp + this_dir + common_cmd2 + log_path + common_cmd3)
    print(name, " finished!")