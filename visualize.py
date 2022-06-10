import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
import torch

def plot_attention_map(variables, labels, attention_mat, save_file, need_labels=True):
    
    # variables: protein
    # labels: drug
    # df = pd.DataFrame(drug_attention, columns=variables, index=labels)
    df = pd.DataFrame(attention_mat, columns=variables, index=labels)

    # fig = plt.figure()
    fig, ax = plt.subplots(figsize=[10, 10], dpi=600)
    # fig, ax = plt.add_subplot(dpi=600)

    cax = ax.matshow(df, interpolation='nearest', cmap='viridis')
    fig.colorbar(cax)

    if need_labels is True:
        tick_spacing = 1
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

        ax.set_xticklabels([''] + list(df.columns))
        ax.set_yticklabels([''] + list(df.index))

        # plt.show()
        # plt.savefig("visualize_attention/test_attention_map", dpi=fig.dpi)
                    # pad_inches=0, bbox_inches="tight")
    plt.savefig(save_file, dpi=fig.dpi, pad_inches=0.5)

def load_seq():
    attention_mat = np.load("visualize_attention/attention_mat.npy")
    attention_mat = np.squeeze(attention_mat, axis=0)
    attention_mat = np.mean(attention_mat, axis=0)

    with open("data/tokenize_data/test.tokenize", "r") as f:
        first_data = f.readlines()[0]
        data = json.loads(first_data)
        seq = data["seq"].split()
    return seq, attention_mat


def compute_site(seq):
    sep_index = 13
    # protein: https://www.uniprot.org/uniprot/P68403
    # results [90, 196]
    begin = 186
    end = 466
    real_seq = ""
    position = []
    pre_fix_len = 0
    for i, seq_i in enumerate(seq):
        # pre_fix_len += len
        # if seq[i] == "[SEP]" and seq[i+1] != "[SEP]":
        #     sep_index = i
        if i > sep_index:       
            real_seq = real_seq + seq_i
            if len(real_seq) >= begin and len(real_seq) <= end:
                position.append(i)
    
    print(position)            
            
    
    
    
    

if __name__ == '__main__':
    seq, attention_mat = load_seq()
    
    # 
    # plot_attention_map(seq, seq, attention_mat, "visualize_attention/all_attention_map.pdf", need_labels=False)
    
    # drug 
    # plot_attention_map(seq[:13], seq[:13],attention_mat[:13, :13], "visualize_attention/drug_attention_map.png")
    
    # compute site
    # compute_site(seq)
    
    # interaction site
    site_range = [181,196]
    plot_attention_map(seq[site_range[0]:site_range[-1]], seq[:13], attention_mat[:13, site_range[0]:site_range[-1]], 
                       "visualize_attention/90_196/drug_attention_map_{}_{}.png".format(site_range[0], site_range[-1]))

    






