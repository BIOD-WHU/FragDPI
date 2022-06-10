import numpy as np
import pandas as pd
import torch
from torch.utils import data
import json
import collections
from torch.utils.data import DataLoader
from subword_nmt.apply_bpe import BPE
import codecs
from tqdm import tqdm
import math
import random
from torch.nn.utils.rnn import pad_sequence


# vocab_path = './ESPF/protein_codes_uniprot.txt'
# bpe_codes_protein = codecs.open(vocab_path)
# pbpe = BPE(bpe_codes_protein, merges=-1, separator='')
# sub_csv = pd.read_csv('./ESPF/subword_units_map_uniprot.csv')
#
# idx2word_p = sub_csv['index'].values
# words2idx_p = dict(zip(idx2word_p, range(0, len(idx2word_p))))

# vocab_path = './ESPF/drug_codes_chembl.txt'
# bpe_codes_drug = codecs.open(vocab_path)
# dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
# sub_csv = pd.read_csv('./ESPF/subword_units_map_chembl.csv')
#
# idx2word_d = sub_csv['index'].values
# words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

# max_d = 205
# max_p = 545


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


# def protein2emb_encoder(x, words2idx_p):
#     max_p = 152
#     # t1 = pbpe.process_line(x).split()  # split
#     t1 = x.split(',')
#     try:
#         i1 = np.asarray([words2idx_p[i] for i in t1])  # index
#     except:
#         i1 = np.array([0])
#         # print(x)
#
#     l = len(i1)
#
#     if l < max_p:
#         i = np.pad(i1, (0, max_p - l), 'constant', constant_values=0)
#         input_mask = ([1] * l) + ([0] * (max_p - l))
#     else:
#         i = i1[:max_p]
#         input_mask = [1] * max_p
#
#     return i, np.asarray(input_mask)


# def drug2emb_encoder(x, dbpe, words2idx_d):
#     max_d = 50
#     # max_d = 100
#     t1 = dbpe.process_line(x)
#     t1 = t1.split()  # split
#     try:
#         i1 = np.asarray([words2idx_d[i] for i in t1])  # index
#     except:
#         i1 = np.array([0])
#         # print(x)
#
#     l = len(i1)
#     print(i1)
#
#     if l < max_d:
#         i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
#         input_mask = ([1] * l) + ([0] * (max_d - l))
#
#     else:
#         i = i1[:max_d]
#         input_mask = [1] * max_d
#
#     return i, np.asarray(input_mask)


def seq2emb_encoder(input_seq, max_len, vocab):
    try:
        ids = np.asarray([vocab[i] for i in input_seq])
    except:
        ids = np.array([0])

    l = len(ids)

    if l < max_len:
        ids = np.pad(ids, (0, max_len - l), 'constant', constant_values=0)
        input_mask = np.array(([1] * l) + ([0] * (max_len - l)))
    else:
        ids = ids[:max_len]
        input_mask = np.array([1] * max_len)

    return ids, input_mask


def seq2emb_encoder_simple(input_seq, max_len, vocab):
    try:
        ids = np.asarray([vocab[i] for i in input_seq])
    except:
        ids = np.array([0])

    # l = len(ids)
    #
    # if l < max_len:
    #     ids = np.pad(ids, (0, max_len - l), 'constant', constant_values=0)
    #     input_mask = np.array(([1] * l) + ([0] * (max_len - l)))
    # else:
    #     ids = ids[:max_len]
    #     input_mask = np.array([1] * max_len)

    return ids


class Data_Encoder(data.Dataset):
    def __init__(self, train_file, tokenizer_config):
        'Initialization'
        # load data
        with open(train_file["sps"], 'r') as f:
            self.sps = f.readlines()
        with open(train_file["smile"], 'r') as f:
            self.smile = f.readlines()
        with open(train_file["affinity"], 'r') as f:
            self.affinity = f.readlines()
        # define tokenizer
        self.begin_id = tokenizer_config["begin_id"]
        self.sep_id = tokenizer_config["separate_id"]
        self.max_len = tokenizer_config["max_len"]
        self.vocab = load_vocab(tokenizer_config["vocab_file"])
        bpe_codes_drug = codecs.open(tokenizer_config["vocab_pair"])
        self.dbpe = BPE(bpe_codes_drug, merges=-1, separator='')

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.sps)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        # tokenization
        d = self.dbpe.process_line(self.smile[index].strip()).split()
        p = self.sps[index].strip().split(',')
        y = np.float(self.affinity[index].strip())

        input_seq = [self.begin_id] + d + [self.sep_id] + p + [self.sep_id]
        token_type_ids = np.concatenate((np.zeros((len(d) + 2), dtype=np.int), np.ones((len(p) + 1), dtype=np.int)))
        token_type_ids = np.pad(token_type_ids, (0, self.max_len - len(input_seq)), 'constant', constant_values=0)
        input, input_mask = seq2emb_encoder(input_seq, self.max_len, self.vocab)
        return torch.from_numpy(input).long(), torch.from_numpy(token_type_ids).long(), torch.from_numpy(
            input_mask).long(), y
        # return len(d), len(p)


class Data_Encoder_mol(data.Dataset):
    def __init__(self, train_file, tokenizer_config):
        'Initialization'
        # load data
        # with open(train_file["sps"], 'r') as f:
        #     self.sps = f.readlines()
        with open(train_file['seq'], 'r') as f:
            self.seq = f.readlines()
        with open(train_file["smile"], 'r') as f:
            self.smile = f.readlines()
        with open(train_file["affinity"], 'r') as f:
            self.affinity = f.readlines()
        # define tokenizer
        self.begin_id = tokenizer_config["begin_id"]
        self.sep_id = tokenizer_config["separate_id"]
        self.max_len = tokenizer_config["max_len"]
        self.vocab = load_vocab(tokenizer_config["vocab_file"])

        bpe_codes_drug = codecs.open(tokenizer_config["vocab_pair"])
        self.dbpe = BPE(bpe_codes_drug, merges=-1, separator='')

        bpe_codes_prot = codecs.open(tokenizer_config["vocab_pair_p"])
        self.pbpe = BPE(bpe_codes_prot, merges=-1, separator='')

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.smile)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        # tokenization
        d = self.dbpe.process_line(self.smile[index].strip()).split()
        p = self.pbpe.process_line(self.seq[index].strip()).split()
        y = np.float64(self.affinity[index].strip())

        input_seq = [self.begin_id] + d + [self.sep_id] + p + [self.sep_id]
        token_type_ids = np.concatenate((np.zeros((len(d) + 2), dtype=np.int), np.ones((len(p) + 1), dtype=np.int)))
        if len(input_seq) > self.max_len:
            input_seq = input_seq[:self.max_len-1] + [self.sep_id]
            token_type_ids = token_type_ids[:self.max_len]
        else:
            token_type_ids = np.pad(token_type_ids, (0, self.max_len - len(input_seq)), 'constant', constant_values=0)
        input, input_mask = seq2emb_encoder(input_seq, self.max_len, self.vocab)
        return torch.from_numpy(input).long(), torch.from_numpy(token_type_ids).long(), torch.from_numpy(input_mask).long(), y
        # return len(d), len(p)


class Data_Encoder_LM(data.Dataset):
    def __init__(self, train_file, tokenizer_config):
        'Initialization'
        # load data
        # with open(train_file["sps"], 'r') as f:
        #     self.sps = f.readlines()
        with open(train_file['seq'], 'r') as f:
            self.seq = f.readlines()
        with open(train_file["smile"], 'r') as f:
            self.smile = f.readlines()
        with open(train_file["affinity"], 'r') as f:
            self.affinity = f.readlines()
        # define tokenizer
        self.begin_id = tokenizer_config["begin_id"]
        self.sep_id = tokenizer_config["separate_id"]
        self.max_len = tokenizer_config["max_len"]
        self.vocab = load_vocab(tokenizer_config["vocab_file"])

        bpe_codes_drug = codecs.open(tokenizer_config["vocab_pair"])
        self.dbpe = BPE(bpe_codes_drug, merges=-1, separator='')

        bpe_codes_prot = codecs.open(tokenizer_config["vocab_pair_p"])
        self.pbpe = BPE(bpe_codes_prot, merges=-1, separator='')

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.smile)


    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        # tokenization
        d = self.dbpe.process_line(self.smile[index].strip()).split()
        p = self.pbpe.process_line(self.seq[index].strip()).split()
        # mask_d, mask_d_posi = self.random_mask(d)
        # mask_p, mask_p_posi = self.random_mask(p)
        y = np.float64(self.affinity[index].strip())
        #
        # input_seq = [self.begin_id] + mask_d + [self.sep_id] + mask_p + [self.sep_id]
        # mask_posi = np.concatenate((np.zeros(1), mask_d_posi, np.zeros(1), mask_p_posi, np.zeros(1)))
        # token_type_ids = np.concatenate((np.zeros((len(d) + 2), dtype=np.int), np.ones((len(p) + 1), dtype=np.int)))
        # if len(input_seq) > self.max_len:
        #     input_seq = input_seq[:self.max_len-1] + [self.sep_id]
        #     token_type_ids = token_type_ids[:self.max_len]
        #     mask_posi = mask_posi[:self.max_len]
        # else:
        #     mask_posi = np.pad(mask_posi, (0, self.max_len - len(input_seq)), 'constant', constant_values=0)
        #     token_type_ids = np.pad(token_type_ids, (0, self.max_len - len(input_seq)), 'constant', constant_values=0)
        # input, input_mask = seq2emb_encoder(input_seq, self.max_len, self.vocab)
        # return torch.from_numpy(input).long(), torch.from_numpy(token_type_ids).long(), torch.from_numpy(input_mask).long(), y, torch.from_numpy(mask_posi).long()
        return " ".join(d), " ".join(p), y
        # return len(d), len(p)

class Data_Provide(data.Dataset):
    def __init__(self, train_file, mask_file):
        'Initialization'
        # load data

        with open(train_file, 'r') as f:
            self.seq = f.readlines()
        with open(mask_file, 'r') as f:
            self.seq_mask = f.readlines()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.seq)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        item = json.loads(self.seq[index])
        mask_item = json.loads(self.seq_mask[index])
        seq = item["seq"]
        seq_mask = mask_item["seq"]
        y = np.float64(item["affinity"])
        return seq, seq_mask, y

class Data_Gen(data.Dataset):
    def __init__(self, train_file):
        'Initialization'
        # load data
        with open(train_file, 'r') as f:
            self.seq = f.readlines()
        # with open(mask_file, 'r') as f:
        #     self.seq_mask = f.readlines()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.seq)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        item = json.loads(self.seq[index])
        # mask_item = json.loads(self.seq_mask[index])
        seq = item["seq"]
        # seq_mask = mask_item["seq"]
        if "affinity" not in item.keys():
            return seq
        else:
            y = np.float64(item["affinity"])
            return seq, y


def get_task(task_name):
    tokenizer_config = {"vocab_file": './config/vocab.txt',
                        "vocab_pair": './config/drug_codes_chembl.txt',
                        "begin_id": '[CLS]',
                        "separate_id": "[SEP]",
                        "max_len": 512
                        }

    if task_name.lower() == 'train':
        df_train = {"sps": './data/train/train_sps',
                    "smile": './data/train/train_smile',
                    "affinity": './data/train/train_ic50',
                    }

        return df_train, tokenizer_config

    elif task_name.lower() == 'test':
        
        df_test = {"sps": './data/test/test_sps',
                   "smile": './data/test/test_smile',
                   "affinity": './data/test/test_ic50',
                   }

        return df_test, tokenizer_config
    
    elif task_name.lower() == 'test_ori_er':
        
        df_test = {"sps": './data/ER/ER_sps',
                   "smile": './data/ER/ER_smile',
                   "affinity": './data/ER/ER_ic50',
                   }

        return df_test, tokenizer_config
    elif task_name.lower() == 'test_ori_gpcr':
        
        df_test = {"sps": './data/GPCR/GPCR_sps',
                   "smile": './data/GPCR/GPCR_smile',
                   "affinity": './data/GPCR/GPCR_ic50',
                   }

        return df_test, tokenizer_config
    elif task_name.lower() == 'test_ori_channel':
        
        df_test = {"sps": './data/Ion_channel/channel_sps',
                   "smile": './data/Ion_channel/channel_smile',
                   "affinity": './data/Ion_channel/channel_ic50',
                   }

        return df_test, tokenizer_config
    elif task_name.lower() == 'test_ori_kinase':
        
        df_test = {"sps": './data/Tyrosine_kinase/kinase_sps',
                   "smile": './data/Tyrosine_kinase/kinase_smile',
                   "affinity": './data/Tyrosine_kinase/kinase_ic50',
                   }

        return df_test, tokenizer_config
    

    elif task_name.lower() == 'train_mol':
        df_train = "data/tokenize_data/train.tokenize"

        tokenizer_config = {"vocab_file": './config/vocab_mol.txt',
                            "vocab_pair": './config/drug_codes_chembl.txt',
                            "vocab_pair_p": './config/protein_codes_uniprot.txt',
                            "begin_id": '[CLS]',
                            "separate_id": "[SEP]",
                            "max_len": 512
                            }

        return df_train, tokenizer_config

    elif task_name.lower() == 'test_mol':
        df_test = "data/tokenize_data/test.tokenize"

        tokenizer_config = {"vocab_file": './config/vocab_mol.txt',
                            "vocab_pair": './config/drug_codes_chembl.txt',
                            "vocab_pair_p": './config/protein_codes_uniprot.txt',
                            "begin_id": '[CLS]',
                            "separate_id": "[SEP]",
                            "max_len": 512
                            }

        return df_test, tokenizer_config
    
    elif task_name.lower() == 'test_er':
        df_test = "data/tokenize_data/er.tokenize"

        tokenizer_config = {"vocab_file": './config/vocab_mol.txt',
                            "vocab_pair": './config/drug_codes_chembl.txt',
                            "vocab_pair_p": './config/protein_codes_uniprot.txt',
                            "begin_id": '[CLS]',
                            "separate_id": "[SEP]",
                            "max_len": 512
                            }
        return df_test, tokenizer_config

    elif task_name.lower() == 'test_gpcr':
        df_test = "data/tokenize_data/gpcr.tokenize"

        tokenizer_config = {"vocab_file": './config/vocab_mol.txt',
                            "vocab_pair": './config/drug_codes_chembl.txt',
                            "vocab_pair_p": './config/protein_codes_uniprot.txt',
                            "begin_id": '[CLS]',
                            "separate_id": "[SEP]",
                            "max_len": 512
                            }
        return df_test, tokenizer_config

    elif task_name.lower() == 'test_channel':
        df_test = "data/tokenize_data/channel.tokenize"

        tokenizer_config = {"vocab_file": './config/vocab_mol.txt',
                            "vocab_pair": './config/drug_codes_chembl.txt',
                            "vocab_pair_p": './config/protein_codes_uniprot.txt',
                            "begin_id": '[CLS]',
                            "separate_id": "[SEP]",
                            "max_len": 512
                            }

        return df_test, tokenizer_config

    elif task_name.lower() == 'test_kinase':
        df_test = "data/tokenize_data/kinase.tokenize"

        tokenizer_config = {"vocab_file": './config/vocab_mol.txt',
                            "vocab_pair": './config/drug_codes_chembl.txt',
                            "vocab_pair_p": './config/protein_codes_uniprot.txt',
                            "begin_id": '[CLS]',
                            "separate_id": "[SEP]",
                            "max_len": 512
                            }

        return df_test, tokenizer_config

    elif task_name.lower() == 'pre-train':
        df_train_mask = "data/tokenize_data/train.tokenize.mask"
        df_train = "data/tokenize_data/train.tokenize"

        tokenizer_config = {"vocab_file": './config/vocab_mol.txt',
                            "vocab_pair": './config/drug_codes_chembl.txt',
                            "vocab_pair_p": './config/protein_codes_uniprot.txt',
                            "begin_id": '[CLS]',
                            "separate_id": "[SEP]",
                            "max_len": 512
                            }

        return df_train, df_train_mask, tokenizer_config
    elif task_name.lower() == 'test-pre-train':
        df_train_mask = "data/tokenize_data/test.tokenize.mask"
        df_train = "data/tokenize_data/test.tokenize"

        tokenizer_config = {"vocab_file": './config/vocab_mol.txt',
                            "vocab_pair": './config/drug_codes_chembl.txt',
                            "vocab_pair_p": './config/protein_codes_uniprot.txt',
                            "begin_id": '[CLS]',
                            "separate_id": "[SEP]",
                            "max_len": 512
                            }

        return df_train, df_train_mask, tokenizer_config

    elif task_name.lower() == 'test-pre-train-er':
        df_train_mask = "data/tokenize_data/er.tokenize.mask"
        df_train = "data/tokenize_data/er.tokenize"

        tokenizer_config = {"vocab_file": './config/vocab_mol.txt',
                            "vocab_pair": './config/drug_codes_chembl.txt',
                            "vocab_pair_p": './config/protein_codes_uniprot.txt',
                            "begin_id": '[CLS]',
                            "separate_id": "[SEP]",
                            "max_len": 512
                            }

        return df_train, df_train_mask, tokenizer_config
    elif task_name.lower() == 'test-pre-train-gpcr':
        df_train_mask = "data/tokenize_data/gpcr.tokenize.mask"
        df_train = "data/tokenize_data/gpcr.tokenize"

        tokenizer_config = {"vocab_file": './config/vocab_mol.txt',
                            "vocab_pair": './config/drug_codes_chembl.txt',
                            "vocab_pair_p": './config/protein_codes_uniprot.txt',
                            "begin_id": '[CLS]',
                            "separate_id": "[SEP]",
                            "max_len": 512
                            }

        return df_train, df_train_mask, tokenizer_config
    elif task_name.lower() == 'test-pre-train-channel':
        df_train_mask = "data/tokenize_data/channel.tokenize.mask"
        df_train = "data/tokenize_data/channel.tokenize"

        tokenizer_config = {"vocab_file": './config/vocab_mol.txt',
                            "vocab_pair": './config/drug_codes_chembl.txt',
                            "vocab_pair_p": './config/protein_codes_uniprot.txt',
                            "begin_id": '[CLS]',
                            "separate_id": "[SEP]",
                            "max_len": 512
                            }

        return df_train, df_train_mask, tokenizer_config
    elif task_name.lower() == 'test-pre-train-kinase':
        df_train_mask = "data/tokenize_data/kinase.tokenize.mask"
        df_train = "data/tokenize_data/kinase.tokenize"

        tokenizer_config = {"vocab_file": './config/vocab_mol.txt',
                            "vocab_pair": './config/drug_codes_chembl.txt',
                            "vocab_pair_p": './config/protein_codes_uniprot.txt',
                            "begin_id": '[CLS]',
                            "separate_id": "[SEP]",
                            "max_len": 512
                            }

        return df_train, df_train_mask, tokenizer_config
    
    
    elif task_name.lower() == 'case_study':
        # df_train_mask = "data/tokenize_data/kinase.tokenize.mask"
        df_train = "case_study/spike.tokenize"

        tokenizer_config = {"vocab_file": './config/vocab_mol.txt',
                            "vocab_pair": './config/drug_codes_chembl.txt',
                            "vocab_pair_p": './config/protein_codes_uniprot.txt',
                            "begin_id": '[CLS]',
                            "separate_id": "[SEP]",
                            "max_len": 512
                            }
        return df_train, tokenizer_config
        
        
        
def random_mask(input_seq, mask_proportion=0.15):
    input = [i.split() for i in input_seq]
    mask_len = [math.ceil(len(i)*mask_proportion) for i in input]
    # mask_posi = np.arange(len(input_seq))
    # mask_token_posi = random.sample(mask_posi, mask_len)
    mask_token_posi = [np.random.choice(len(i), j) for i, j in zip(input, mask_len)]
    # mask_vec = np.zeros(len(input_seq))
    for i, posi in enumerate(mask_token_posi):
        for j in posi:
            choice = random.random()
            if choice < 0.8:
                input[i][j] = "[MASK]"
            # mask_vec[i] = 1
    # elif choice >= 0.8 and choice < 0.9:

    return input


class Tokenizer(object):
    def  __init__(self, tokenizer_config):
        self.begin_id = tokenizer_config["begin_id"]
        self.sep_id = tokenizer_config["separate_id"]
        self.max_len = tokenizer_config["max_len"]
        self.vocab = load_vocab(tokenizer_config["vocab_file"])

    def seq2emb_encoder_simple(self, input_seq, vocab):
        all_ids = []
        for i in input_seq:
            try:
                id = vocab[i]
                all_ids.append(id)
            except:
                id = vocab["[UNK]"]
                all_ids.append(id)
        ids = np.asarray(all_ids)
        return ids

    def convert_token_to_ids(self, seq):
        # input_seq = [[self.begin_id] + i + [self.sep_id] + j + [self.sep_id] for i, j in zip(mask_d, mask_p)]
        # input_seq_ori = [[self.begin_id] + i.split() + [self.sep_id] + j.split() + [self.sep_id] for i, j in zip(d, p)]
        # mask_posi = np.concatenate((np.zeros(1), mask_d_posi, np.zeros(1), mask_p_posi, np.zeros(1)))
        # token_type_ids = [[np.concatenate((np.zeros((len(d) + 2), dtype=np.int), np.ones((len(p) + 1), dtype=np.int)))] for d, p in zip(mask_d, mask_p)]
        # seq = seq.split()
        all_seq = [i.split() for i in seq]
        for i, seq_i in enumerate(all_seq):
            if len(seq_i) > self.max_len:
                all_seq[i] = seq_i[:self.max_len-1] + [self.sep_id]
                # input_seq_ori[i] = seq[:self.max_len-1] + [self.sep_id]
            # token_type_ids = token_type_ids[:self.max_len]
            # mask_posi = mask_posi[:self.max_len]
        # else:
            # mask_posi = np.pad(mask_posi, (0, self.max_len - len(input_seq)), 'constant', constant_values=0)
            # token_type_ids = np.pad(token_type_ids, (0, self.max_len - len(input_seq)), 'constant', constant_values=0)
        all_seq_ids = []
        # all_seq_ori = []
        # all_mask = []
        for seq in all_seq:
            input = self.seq2emb_encoder_simple(seq, self.vocab)
            # input_ori = seq2emb_encoder_simple(ori, self.max_len, self.vocab)
            all_seq_ids.append(torch.from_numpy(input).long())
            # all_seq_ori.append(torch.from_numpy(input_ori).long())

        input = pad_sequence(all_seq_ids, batch_first=True)
        # input_ori = pad_sequence(all_seq_ori, batch_first=True)
        input_mask = (input != 0).long()
        # input_mask = pad_sequence(all_mask)
        # return torch.from_numpy(input).long(), torch.from_numpy(input_mask).long(), torch.from_numpy(token_type_ids).long()
        # return input, input_mask, input_ori
        return input, input_mask



if __name__ == "__main__":
    # local test
    # dataFolder = './IC50/SPS/train_smile'
    # with open(dataFolder, 'r') as f:
    #     train_smi = f.readlines()
    # drug_smi = train_smi[0]
    # d_v, input_mask_d = drug2emb_encoder(drug_smi)

    # test load vocab
    # vocab_file = './ESPF/vocab.txt'
    # vocab = load_vocab(vocab_file)

    # test train
    task = 'pre-train'
    data_file, data_mask, tokenizer_config = get_task(task)
    dataset = Data_Provide(data_file, data_mask)
    tokenizer = Tokenizer(tokenizer_config)
    data_loder_para = {'batch_size': 2,
                       'shuffle': False,
                       'num_workers': 0,
                       }
    data_generator = DataLoader(dataset, **data_loder_para)
    all_len = []
    m = 0
    for i, (seq, seq_mask, affinity) in enumerate(tqdm(data_generator)):
        input_random_mask, attention_mask = tokenizer.convert_token_to_ids(seq_mask)
        label, _ = tokenizer.convert_token_to_ids(seq)
        posi = torch.where(input_random_mask == 1)
        target = label[posi]
        a = input_random_mask == 4
        if torch.sum(a) > 2:
            print(torch.sum(a))
        # a = seq[0].split()
        # b = seq_mask[0].split()
        # all_len.append(len(a))
        # if len(a) > 512:
        #     m += 1
        # if len(a) != len(b):
        #     print(seq)
        #     print(i)
    # all_len = np.array(all_len)
    # print(np.max(all_len))
    # print(np.mean(all_len))
    # print(m)
