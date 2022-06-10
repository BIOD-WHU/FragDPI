import numpy as np
from tqdm import tqdm

def z_score(data, save, enlarge):
    with open(data, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        aff = np.float64(line.strip())
        data.append(aff)
    data = np.array(data)
    ave = np.mean(data)
    std = np.std(data)
    new_affinity = (data - ave) / std
    new_affinity *= enlarge
    new_affinity = list(new_affinity)
    with open(save, 'w') as f:
        for aff in tqdm(new_affinity):
            f.write(str(aff) + '\n')

def reform(input_file_path, result_save_path, average, std, enlarge):
    with open(input_file_path, 'r') as f:
        res = f.readlines()
    with open(result_save_path, 'w') as f:
        for line in tqdm(res):
            data = float(line.strip())
            ori = ((data / enlarge) * std) + average
            f.write(str(ori) + '\n')




if __name__ == '__main__':
    average = 6.339674062480976
    std = 1.4751794034241978

    # gengerate z-score dataset
    # data = '../data/train_ic50'
    # save = '../data/train_z_1_ic50'
    # enlarge = 1
    # z_score(data, save, enlarge)

    # reform result
    result = '../predict/lr-1e-5-batch-32-e-10-layer3-0503-z-1-step-82370/test_1.txt'
    save = '../predict/lr-1e-5-batch-32-e-10-layer3-0503-z-1-step-82370/test.txt'
    reform(result, save, average, std, 1)