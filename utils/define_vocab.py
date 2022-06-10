import pandas as pd
import numpy as np

sub_csv = pd.read_csv('../config/subword_units_map_chembl.csv')
idx2word_d = sub_csv['index'].values

sub_csv = pd.read_csv('../config/subword_units_map_uniprot.csv')
idx2word_p = sub_csv['index'].values
# words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

spqcial_tokens = np.array(['[PAD]', '[MASK]', '[CLS]', '[SEP]', '[UNK]', '[unused1]', '[unused2]', '[unused3]', '[unused4]', '[unused5]'])

all_tokens = np.concatenate((spqcial_tokens, idx2word_p, idx2word_d))

save = '../config/vocab_mol.txt'

with open(save, 'w') as f:
    for token in all_tokens:
        f.write(str(token) + '\n')

