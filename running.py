import subprocess
import os
base_dir = "/home/gaoyifei/RNASSR/"

RBP_list = ['ALKBH5_Baltz2012', 'C17ORF85_Baltz2012', 'C22ORF28_Baltz2012', 'CAPRIN1_Baltz2012',
            'CLIPSEQ_ELAVL1', 'CLIPSEQ_SFRS1', 'ICLIP_HNRNPC', 'ICLIP_TIA1',
            'PARCLIP_ELAVL1', 'PARCLIP_ELAVL1A', 'PARCLIP_EWSR1',
            'PARCLIP_IGF2BP123', 'PARCLIP_MOV10_Sievers', 'PARCLIP_PUM2',
            'PARCLIP_QKI', 'PARCLIP_TAF15', 'ZC3H7B_Baltz2012']

datadir = "/data/gaoyifei"
for rbp in RBP_list:
    if rbp in ['CAPRIN1_Baltz2012', 'PARCLIP_IGF2BP123', 'PARCLIP_MOV10_Sievers', 'ZC3H7B_Baltz2012',
                'C22ORF28_Baltz2012', 'PARCLIP_ELAVL1A', 'PARCLIP_TAF15', 'PARCLIP_FUS', 'PARCLIP_EWSR1',
                'PARCLIP_HUR', 'PARCLIP_PUM2', 'PARCLIP_AGO1234', 'ALKBH5_Baltz2012',
                'C17ORF85_Baltz2012', 'PARCLIP_QKI', 'PARCLIP_ELAVL1', 'CLIPSEQ_SFRS1', 'CLIPSEQ_AGO2',
                'CLIPSEQ_ELAVL1']:
        debias = False
        path_template = os.path.join(datadir, 'data', 'GraphProt_CLIP_sequences', 'RNAGraphProb', rbp+'.pkl')
    elif rbp in ['ICLIP_HNRNPC', 'ICLIP_TDP43', 'ICLIP_TIA1', 'ICLIP_TIAL1', 'PTBv1']:
        debias = True
        path_template = os.path.join(datadir, 'data', 'GraphProt_CLIP_sequences', 'RNAGraphProb_debias', rbp + '.pkl')
    else:
        raise ValueError('Warning, %s is not a valid rbp!' % (rbp))
    if os.path.exists(path_template) is True:
        print(rbp + '.pkl' + " already exists!")
    else:
        command = "/home/gaoyifei/.conda/envs/RNASSR/bin/python preprocess.py \""+rbp+"\" " + str(debias)+" --probablistic False"
        subprocess.call(f"{command}" , shell=True)
        print(rbp, "finished!")


