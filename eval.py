import numpy as np
import re


def eval_result(pred, label):
    pred = np.array(pred)
    label = np.array(label)
    num = len(pred)
    diff = pred - label
    mse = np.sum(np.power(diff, 2)) / num
    rmse = np.sqrt(mse)
    pearson_co = np.corrcoef(pred, label)

    return rmse, pearson_co


def eval(pred_path, label_path):
    with open(pred_path, 'r') as f:
        pred = f.readlines()
        pred = [float(i.strip()) for i in pred]
    with open(label_path, 'r') as f:
        label = f.readlines()
        label = [float(i.strip()) for i in label]
    remse, r_mat = eval_result(pred, label)
    r = r_mat[0, 1]
    file = pred_path.split("/")[-1]
    save_path = pred_path.replace(file, 'eval_results')
    with open(save_path, 'w') as f:
        f.write('RMSE : {} ; Pearson Correlation Coefficient : {}'.format(remse, r))
    print('RMSE : {} ; Pearson Correlation Coefficient : {}'.format(remse, r))


if __name__ == '__main__':
    # with open('pre_test.sh', 'r') as f:
    #     pred_dir = f.readline()
    #     pred_dir = pred_dir.split()[5].split('/')[-1]
    # pred_result = './predict/{}/test.txt'.format(pred_dir)
    # pred_result = './predict/add_pretrain_1019-s-329480_v2/test_mol.txt'
    # pred_result = './predict/add_pretrain_1019-s-329480-er/test_mol.txt'

    # eval single file
    # pred_file = "./predict/without-pre-train-layer-6-1021-s-988440-test/test_mol.txt"
    # test_label_path = './data/test/test_ic50'
    # eval(pred_file, test_label_path)



    # eval all

    test_label_path = './data/test/test_ic50'
    test_label_path_ER = './data/ER/ER_ic50'
    test_label_path_GPCR = './data/GPCR/GPCR_ic50'
    test_label_path_Ion_channel = './data/Ion_channel/channel_ic50'
    test_label_path_Tyrosine_kinase = './data/Tyrosine_kinase/kinase_ic50'

    # test mol
    # pred_test = "./predict/without-pre-train-layer-6-1021-s-988440-test/test_mol.txt"
    # er = "./predict/without-pre-train-layer-6-1021-s-988440-er/test_er.txt"
    # gpcr = "./predict/without-pre-train-layer-6-1021-s-988440-gpcr/test_gpcr.txt"
    # channel = "./predict/without-pre-train-layer-6-1021-s-988440-channel/test_channel.txt"
    # kinase = "./predict/without-pre-train-layer-6-1021-s-988440-kinase/test_kinase.txt"
    
    # test 
    # pred_test = "predict/train_ori_1217-s-296532/test.txt"
    # er = "predict/train_ori_1217-s-296532/test_ori_er.txt"
    # gpcr = "predict/train_ori_1217-s-296532/test_ori_gpcr.txt"
    # channel = "predict/train_ori_1217-s-296532/test_ori_channel.txt"
    # kinase = "predict/train_ori_1217-s-296532/test_ori_kinase.txt"
    
    # deepdta
    # pred_test = "baselines/DeepDTA/source/output/test/results.txt"
    # er = "baselines/DeepDTA/source/output/ER/results.txt"
    # gpcr = "baselines/DeepDTA/source/output/GPCR/results.txt"
    # channel = "baselines/DeepDTA/source/output/Ion_channel/results.txt"
    # kinase = "baselines/DeepDTA/source/output/Tyrosine_kinase/results.txt"
    
    
    # attentiondta
    pred_test = "baselines/AttentionDTA_BIBM/results/test/test.txt"
    er = "baselines/AttentionDTA_BIBM/results/ER/test.txt"
    gpcr = "baselines/AttentionDTA_BIBM/results/GPCR/test.txt"
    channel = "baselines/AttentionDTA_BIBM/results/channel/test.txt"
    kinase = "baselines/AttentionDTA_BIBM/results/kinase/test.txt"


    pred_list = [pred_test, er, gpcr, channel, kinase]
    label_list = [test_label_path, test_label_path_ER, test_label_path_GPCR, test_label_path_Ion_channel, test_label_path_Tyrosine_kinase]
    for i, j in zip(pred_list, label_list):
        print(i)
        eval(i, j)


    
    

