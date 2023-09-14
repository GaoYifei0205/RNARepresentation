

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
"PARCLIP_AGO1234",
"PARCLIP_ELAVL1",
"PARCLIP_ELAVL1A",
"PARCLIP_EWSR1",
"PARCLIP_FUS",
"PARCLIP_HUR",
"PARCLIP_IGF2BP123",
"PARCLIP_MOV10_Sievers",
"PARCLIP_PUM2",
"PARCLIP_QKI",
"PARCLIP_TAF15",
"PTBv1",
"ZC3H7B_Baltz2012"]

def get_num(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    record_count = 0
    for line in lines:
        if line.startswith('>'):
            record_count += 1

    return record_count



# for rbp in result:
#     base_file_path = '/amax/data/gaoyifei/GraphProt/GraphProt_CLIP_sequences/'+rbp
#     test_neg = base_file_path + '/test/negatives/data.fa'
#     test_pos = base_file_path + '/test/positives/data.fa'
#     train_neg = base_file_path + '/train/negatives/data.fa'
#     train_pos = base_file_path + '/train/positives/data.fa'
#     print(rbp," test neg num: ", get_num(test_neg))
#     print(rbp," test pos num: ", get_num(test_pos))
#     print(rbp," train neg num: ", get_num(train_neg))
#     print(rbp," train pos num: ", get_num(train_pos))

def get_id_from_line(line):
    start = line.find('>') + 1
    end = line.find('.')
    if start > 0 and end > 0:
        return line[start:end]
    return None

id = '>ICLIP_TIAL1.slop15.train.neg_16593;chr18,9353058,9353133,+'
rbp = get_id_from_line(id)
if(rbp):
    print(rbp)