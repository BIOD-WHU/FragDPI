from subword_nmt.apply_bpe import BPE
import codecs
import json
import numpy as np
from tqdm import tqdm
import math
import random

def get_tokenzie_seq(file, save, mask=False):
    begin_token = '[CLS]'
    separate_token = "[SEP]"
    with open(file['seq'], 'r') as f:
        seq = f.readlines()
    with open(file["smile"], 'r') as f:
        smile = f.readlines()
    with open(file["affinity"], 'r') as f:
        affinity = f.readlines()

    bpe_codes_drug = codecs.open('./config/drug_codes_chembl.txt')
    dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
    bpe_codes_prot = codecs.open('./config/protein_codes_uniprot.txt')
    pbpe = BPE(bpe_codes_prot, merges=-1, separator='')

    with open(save, "w") as f:
        for i in tqdm(range(len(seq))):
            d = dbpe.process_line(smile[i].strip()).split()
            p = pbpe.process_line(seq[i].strip()).split()
            if mask == True:
                d = random_mask(d)
                p = random_mask(p)
            final_seq = [begin_token] + d + [separate_token] + p + [separate_token]
            affinity_num = affinity[i].strip()
            item = {
                "seq": " ".join(final_seq),
                "affinity": affinity_num
            }
            new_item = json.dumps(item)
            f.write(new_item + '\n')


def random_mask(input_seq, mask_proportion=0.15):
    mask_len = math.ceil(len(input_seq)*mask_proportion)
    mask_token_posi = np.random.choice(len(input_seq), mask_len)

    for i in mask_token_posi:
        choice = random.random()
        if choice < 0.8:
            input_seq[i] = "[MASK]"
            # mask_vec[i] = 1
    # elif choice >= 0.8 and choice < 0.9:

    return input_seq

def get_tokenzie_seq_case(file, save, mask=False):
    begin_token = '[CLS]'
    separate_token = "[SEP]"
    with open(file['seq'], 'r') as f:
        seq = f.readlines()
        seq = [i.strip() for i in seq]
        seq = "".join(seq)
    with open(file["smile"], 'r') as f:
        smile = f.readlines()
    # with open(file["affinity"], 'r') as f:
    #     affinity = f.readlines()

    bpe_codes_drug = codecs.open('./config/drug_codes_chembl.txt')
    dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
    bpe_codes_prot = codecs.open('./config/protein_codes_uniprot.txt')
    pbpe = BPE(bpe_codes_prot, merges=-1, separator='')

    with open(save, "w") as f:
        for i in tqdm(range(len(smile))):
            d = dbpe.process_line(smile[i].strip()).split()
            p = pbpe.process_line(seq).split()
            if mask == True:
                d = random_mask(d)
                p = random_mask(p)
            final_seq = [begin_token] + d + [separate_token] + p + [separate_token]
            # affinity_num = affinity[i].strip()
            item = {
                "seq": " ".join(final_seq),
                # "affinity": affinity_num
            }
            new_item = json.dumps(item)
            f.write(new_item + '\n')


if __name__ == '__main__':
    # file_train = {"sps": './data/train/train_sps',
    #             'seq': './data/train/train_protein_seq',
    #             "smile": './data/train/train_smile',
    #             "affinity": './data/train/train_ic50',
    #             }
    # save = "./data/tokenize_data/train.tokenize"
    # save_mask = "./data/tokenize_data/train.tokenize.mask"

    df_test = {"sps": './data/test/test_sps',
               'seq': './data/test/test_protein_seq',
               "smile": './data/test/test_smile',
               "affinity": './data/test/test_ic50',
               }
    df_ER = {"sps": './data/ER/ER_sps',
               'seq': './data/ER/ER_protein_seq',
               "smile": './data/ER/ER_smile',
               "affinity": './data/ER/ER_ic50',
               }
    df_GPCR = {"sps": './data/GPCR/GPCR_sps',
               'seq': './data/GPCR/GPCR_protein_seq',
               "smile": './data/GPCR/GPCR_smile',
               "affinity": './data/GPCR/GPCR_ic50',
               }
    df_Ion_channel = {"sps": './data/Ion_channel/channel_sps',
               'seq': './data/Ion_channel/channel_protein_seq',
               "smile": './data/Ion_channel/channel_smile',
               "affinity": './data/Ion_channel/channel_ic50',
               }
    df_Tyrosine_kinase = {"sps": './data/Tyrosine_kinase/kinase_sps',
               'seq': './data/Tyrosine_kinase/kinase_protein_seq',
               "smile": './data/Tyrosine_kinase/kinase_smile',
               "affinity": './data/Tyrosine_kinase/kinase_ic50',
               }
    # save = "./data/tokenize_data/test.tokenize"
    # save = "./data/tokenize_data/test.tokenize.mask"
    # get_tokenzie_seq(df_test, save)
    # get_tokenzie_seq(file_train, save_mask, mask=True)
    # save_er = "./data/tokenize_data/er.tokenize"
    # save_GPCR = "./data/tokenize_data/gpcr.tokenize"
    # save_channel = "./data/tokenize_data/channel.tokenize"
    # save_kinase = "./data/tokenize_data/kinase.tokenize"
    save_er_mask = "./data/tokenize_data/er.tokenize.mask"
    save_GPCR_mask = "./data/tokenize_data/gpcr.tokenize.mask"
    save_channel_mask = "./data/tokenize_data/channel.tokenize.mask"
    save_kinase_mask = "./data/tokenize_data/kinase.tokenize.mask"
    # get_tokenzie_seq(df_ER, save_er)
    # get_tokenzie_seq(df_GPCR, save_GPCR)
    # get_tokenzie_seq(df_Ion_channel, save_channel)
    # get_tokenzie_seq(df_Tyrosine_kinase, save_kinase)
    # get_tokenzie_seq(df_ER, save_er_mask, mask=True)
    # get_tokenzie_seq(df_GPCR, save_GPCR_mask, mask=True)
    # get_tokenzie_seq(df_Ion_channel, save_channel_mask, mask=True)
    # get_tokenzie_seq(df_Tyrosine_kinase, save_kinase_mask, mask=True)
    
    
    
    
    df_case = {'seq': 'case_study/spike.txt',
               "smile": './data/test/test_smile',
            #    "affinity": './data/Tyrosine_kinase/kinase_ic50',
               }
    save_case = "./case_study/spike.tokenize"
    get_tokenzie_seq_case(df_case, save_case)