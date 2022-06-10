from yaml import load
from dataset import Data_Encoder, get_task, Data_Encoder_mol, Data_Gen, Tokenizer
import torch
from torch.utils.data import DataLoader
from configuration_bert import BertConfig
from modeling_bert import BertAffinityModel
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "5"


    
def load_embedding(data_file):
    tokenizer_config = {"vocab_file": './config/vocab_mol.txt',
                            "vocab_pair": './config/drug_codes_chembl.txt',
                            "vocab_pair_p": './config/protein_codes_uniprot.txt',
                            "begin_id": '[CLS]',
                            "separate_id": "[SEP]",
                            "max_len": 512
                            }
    tokenizer = Tokenizer(tokenizer_config)
    sep_id = 3
    dataset = Data_Gen(data_file)
    data_generator = DataLoader(dataset, batch_size=1, shuffle=False)
    config = BertConfig.from_pretrained('./config/config_layer_6_mol.json')
    model = BertAffinityModel(config)
    model.load_state_dict(torch.load('./model/add_pretrain_1019/epoch-9-step-329480-loss-0.736057146887367.pth'), strict=True)
    

    all_drug = []
    all_protein = []
    
    for i, (input, affinity) in enumerate(data_generator):
        # input = input[1:]
        input_ids, attention_mask = tokenizer.convert_token_to_ids(input)
        input_embs = model.embeddings(input_ids)
        
        sep_index = torch.where(input_ids[:, :-1] == sep_id)[-1]
        drug_emb = input_embs[:, 1:sep_index].squeeze(0).detach().numpy()
        protein_embs = input_embs[:, sep_index+1:-1].squeeze(0).detach().numpy()
        
        all_drug.append(drug_emb)
        all_protein.append(protein_embs)
    
    return all_drug, all_protein    
        

def plot_drug_protein(save):

    drug_embs, protein_embs = load_embedding("add_figure/sample_data/test_sample")
    
    
    all_drug_sub = np.concatenate(drug_embs)
    all_protein_sub = np.concatenate(protein_embs)[:len(all_drug_sub)]
    
    all_data = np.concatenate((all_drug_sub, all_protein_sub))

    y = np.array([0]*len(all_drug_sub) + [1]*len(all_protein_sub))
    
    # t-sne
    
    X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(all_data)
    
    # plot
    fig, ax=plt.subplots(dpi=600)
    plt.axis("off")
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)

    plt.scatter(X_tsne[y==0, 0], X_tsne[y==0, 1], c="darkcyan", s=5, label="Drug", marker='^')
    plt.scatter(X_tsne[y==1, 0], X_tsne[y==1, 1], c="deepskyblue", s=5, label="Protein", marker="s")
    # plt.scatter(X_tsne[y==2, 0], X_tsne[y==2, 1], c="salmon", s=5, label="Story")
    plt.legend(labels=["Drug", "Protein"], loc=1)

    plt.savefig(save, dpi=fig.dpi, pad_inches=0, bbox_inches="tight")



def plot_protein_sub(save):
    
    drug_embs, protein_embs = load_embedding("add_figure/sample_data/test_sample")
    drug_1 = protein_embs[0]
    drug_2 = protein_embs[1]
    drug_3 = protein_embs[2]
    # drug_4 = protein_embs[3]
    y = np.array([0]*len(drug_1) + [1]*len(drug_2) + [2]*len(drug_3)) 
                #  + [3]*len(drug_4))
    
    
    # all_data = np.concatenate((drug_1, drug_2, drug_3, drug_4))
    all_data = np.concatenate((drug_1, drug_2, drug_3))
    
    X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(all_data)
    
    # plot
    fig, ax=plt.subplots(dpi=600)
    plt.axis("off")

    plt.scatter(X_tsne[y==0, 0], X_tsne[y==0, 1], c="darkcyan", s=5, label="PTPH1", marker='^')
    plt.scatter(X_tsne[y==1, 0], X_tsne[y==1, 1], c="deepskyblue", s=5, label="mGluRs", marker="s")
    plt.scatter(X_tsne[y==2, 0], X_tsne[y==2, 1], c="salmon", s=5, label="EZH2")
    # plt.scatter(X_tsne[y==3, 0], X_tsne[y==3, 1], s=5, label="Protein_4")
    # plt.legend(labels=["Protein_1", "Protein_2", "Protein_3", "Protein_4"], loc=1)
    plt.legend(labels=["PTPH1", "mGluRs", "EZH2"], loc=1)

    plt.savefig(save, dpi=fig.dpi, pad_inches=0, bbox_inches="tight")
    
    
    

def plot_drug_sub(save):
    
    drug_embs, protein_embs = load_embedding("add_figure/sample_data/test_sample")
    drug_1 = drug_embs[0]
    drug_2 = drug_embs[1]
    drug_3 = drug_embs[2]
    # drug_4 = protein_embs[3]
    y = np.array([0]*len(drug_1) + [1]*len(drug_2) + [2]*len(drug_3)) 
                #  + [3]*len(drug_4))
    
    
    # all_data = np.concatenate((drug_1, drug_2, drug_3, drug_4))
    all_data = np.concatenate((drug_1, drug_2, drug_3))
    
    X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(all_data)
    
    # plot
    fig, ax=plt.subplots(dpi=600)
    plt.axis("off")

    plt.scatter(X_tsne[y==0, 0], X_tsne[y==0, 1], c="darkcyan", s=5, label="Drug_1", marker='^')
    plt.scatter(X_tsne[y==1, 0], X_tsne[y==1, 1], c="deepskyblue", s=5, label="Drug_2", marker="s")
    plt.scatter(X_tsne[y==2, 0], X_tsne[y==2, 1], c="salmon", s=5, label="Drug_3")
    # plt.scatter(X_tsne[y==3, 0], X_tsne[y==3, 1], s=5, label="Protein_4")
    # plt.legend(labels=["Protein_1", "Protein_2", "Protein_3", "Protein_4"], loc=1)
    plt.legend(labels=["Drug_1", "Drug_2", "Drug_3"], loc=1)

    plt.savefig(save, dpi=fig.dpi, pad_inches=0, bbox_inches="tight")

    

if __name__ == '__main__':
    plot_drug_protein("drug_and_protein_sub")
    # plot_drug_sub("three_drug_sub")
    # plot_protein_sub("three_protein_sub")
