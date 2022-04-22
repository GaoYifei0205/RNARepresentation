import time
import os
import sys
import pickle
from data.RNAGraph import RNAGraphDatasetDGL

start = time.time()
DATASET_NAME = sys.argv[1]
debias = sys.argv[2]
fold_algo = sys.argv[3]
if sys.argv[4] == 'False':
    probabilistic = False
elif sys.argv[4] == 'True':
    probabilistic = True

if DATASET_NAME in ['CAPRIN1_Baltz2012', 'PARCLIP_IGF2BP123', 'PARCLIP_MOV10_Sievers', 'ZC3H7B_Baltz2012',
           'C22ORF28_Baltz2012', 'PARCLIP_ELAVL1A', 'PARCLIP_TAF15', 'PARCLIP_FUS', 'PARCLIP_EWSR1',
           'PARCLIP_HUR', 'PARCLIP_PUM2', 'PARCLIP_AGO1234', 'ALKBH5_Baltz2012',
           'C17ORF85_Baltz2012', 'PARCLIP_QKI', 'PARCLIP_ELAVL1', 'CLIPSEQ_SFRS1', 'CLIPSEQ_AGO2',
           'CLIPSEQ_ELAVL1'] and debias=='True':
    raise ValueError('Warning, %s is not debiased.debias should be set to False!' % (DATASET_NAME))

elif DATASET_NAME in ['ICLIP_HNRNPC', 'ICLIP_TDP43', 'ICLIP_TIA1', 'ICLIP_TIAL1', 'PTBv1'] and debias=='False':
    raise ValueError('Warning, %s is not biased.debias should be set to True!' % (DATASET_NAME))



# os.chdir('../../') # go to root folder of the project
print(os.getcwd())

# basedir = os.getcwd()
basedir = '/data/gaoyifei'
if debias == 'True':
    path_template = os.path.join(basedir, 'data', 'GraphProt_CLIP_sequences', 'RNAGraphProb_debias')
else:
    path_template = os.path.join(basedir, 'data', 'GraphProt_CLIP_sequences', 'RNAGraphProb')
if os.path.exists(path_template) is False:
    os.mkdir(path_template)
path_template = os.path.join(path_template, DATASET_NAME + fold_algo+'.pkl')
# if os.path.exists(path_template) is True:
#     print(DATASET_NAME + '.pkl' + " already exists!")
#     exit()
# with open(path_template, 'wb') as f:
#     pickle.dump([], f)

dataset = RNAGraphDatasetDGL(DATASET_NAME, fold_algo, probabilistic, debias=debias)

print('Time (sec):', time.time() - start)  # 356s=6min

print('length of train dataset: ', len(dataset.train))
print('length of validation dataset: ', len(dataset.val))
print('length of test dataset: ', len(dataset.test))

print('first element in train dataset: ', dataset.train[0])
print('first element in validation dataset: ', dataset.val[0])
print('first element in test dataset: ', dataset.test[0])

start = time.time()

with open(path_template, 'wb') as f:
    pickle.dump([dataset.train, dataset.val, dataset.test], f)

print('Time (sec):', time.time() - start)  # 38s
