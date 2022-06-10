from subword_nmt.apply_bpe import BPE
import codecs
import collections

bpe_codes_drug = codecs.open('../config/drug_codes_chembl.txt')
dbpe = BPE(bpe_codes_drug, merges=-1, separator='')

bpe_codes_prot = codecs.open('../config/protein_codes_uniprot.txt')
pbpe = BPE(bpe_codes_prot, merges=-1, separator='')


def load_file(file):
    data = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data.append(line.strip('\n'))
        return data

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab

def seq2vec(protein, drug):
    start_token = '[CLS]'
    sep_token = '[SEP]'

    prots = load_file(protein)
    drugs = load_file(drug)
    for p, d in zip(prots, drugs):
        d = dbpe.process_line(d).split()
        p = pbpe.process_line(p).split()
        tokens = [start_token] + d + [sep_token] + p + [sep_token]
        print(len(p))


if __name__ == '__main__':
    seq = '../data/test/test_protein_seq'
    simle = '../data/train/train_smile'
    vocab = '../config/vocab_mol.txt'
    seq2vec(seq, simle)