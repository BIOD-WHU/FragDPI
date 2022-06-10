from argparse import ArgumentParser

import numpy as np
from dataset import Data_Encoder, get_task, Data_Encoder_mol, Data_Gen, Tokenizer
import torch
from torch.utils.data import DataLoader
from configuration_bert import BertConfig
from modeling_bert import BertAffinityModel
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
# torch.set_default_tensor_type(torch.DoubleTensor)









def train(args, model, dataset, tokenizer, pre_train=False):
    data_loder_para = {'batch_size': args.batch_size,
                       'shuffle': True,
                       'num_workers': args.workers,
                       }

    data_generator = DataLoader(dataset, **data_loder_para)

    if pre_train == True:
        model.load_state_dict(torch.load(args.init), strict=True)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fct = torch.nn.MSELoss()
    writer = SummaryWriter('./log/' + args.savedir)
    num_step = args.epochs * len(data_generator)
    step = 0
    save_step = num_step // 10
    # detect GPU
    if torch.cuda.is_available():
        model.cuda()
    # print(model)
    print('epoch num : {}'.format(args.epochs))
    print('step num : {}'.format(num_step))
    print('batch size : {}'.format(args.batch_size))
    print('learning rate : {}'.format(args.lr))
    print('begin training')
    # training
    for epoch in range(args.epochs):
        for i, (input_ids, token_type_ids, attention_mask, affinity) in enumerate(data_generator):
            # use cuda
            # input model
            # input_ids, attention_mask = tokenizer.convert_token_to_ids(input)
            pred_affinity = model(input_ids=input_ids.cuda(), token_type_ids=token_type_ids.cuda(), attention_mask=attention_mask.cuda())
            loss = loss_fct(pred_affinity, affinity.cuda().float().unsqueeze(-1))
            step += 1
            writer.add_scalar('loss', loss, global_step=step)
            # Update gradient
            opt.zero_grad()
            loss.backward()
            opt.step()

            #                 if (i % 100 == 0):
            print('Training at Epoch ' + str(epoch + 1) + ' step ' + str(step) + ' with loss ' + str(
                loss.cpu().detach().numpy()))
            # save
            if epoch >= 1 and step % save_step == 0:
                save_path = './model/' + args.savedir + '/'
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                torch.save(model.state_dict(), save_path + 'epoch-{}-step-{}-loss-{}.pth'.format(epoch, step, loss))
    print('training  over')
    writer.close()


def test(args, model, dataset, tokenizer):
    data_loder_para = {'batch_size': args.batch_size,
                       'shuffle': False,
                       'num_workers': args.workers,
                       }
    data_generator = DataLoader(dataset, **data_loder_para)

    with torch.no_grad():
        # if torch.cuda.is_available():
        model.load_state_dict(torch.load(args.init), strict=True)
        model.cuda()
        # else:
        #     model.load_state_dict(torch.load(args.init, map_location=torch.device('cpu')), strict=True)
        model.eval()
        if not os.path.exists(args.output):
            os.mkdir(args.output)
        result = args.output + '/' + '{}.txt'.format(args.task)
        print('begin predicting')
        with open(result, 'w') as f:
            for i, (input_ids, token_type_ids, attention_mask, affinity) in enumerate(tqdm(data_generator)):
                # input_ids, attention_mask = tokenizer.convert_token_to_ids(input)
                pred_affinity = model(input_ids=input_ids.cuda(), token_type_ids=token_type_ids.cuda(), attention_mask=attention_mask.cuda())
                pred_affinity = pred_affinity.cpu().numpy().squeeze(-1)
                for res in pred_affinity:
                    f.write(str(res) + '\n')

    # if args.do_eval:
    #     os.system('python eval.py')



def train_mol(args, model, dataset, tokenizer, pre_train=False):
    data_loder_para = {'batch_size': args.batch_size,
                       'shuffle': True,
                       'num_workers': args.workers,
                       }

    data_generator = DataLoader(dataset, **data_loder_para)

    if pre_train == True:
        model.load_state_dict(torch.load(args.init), strict=True)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fct = torch.nn.MSELoss()
    writer = SummaryWriter('./log/' + args.savedir)
    num_step = args.epochs * len(data_generator)
    step = 0
    save_step = num_step // 10
    # detect GPU
    if torch.cuda.is_available():
        model.cuda()
    # print(model)
    print('epoch num : {}'.format(args.epochs))
    print('step num : {}'.format(num_step))
    print('batch size : {}'.format(args.batch_size))
    print('learning rate : {}'.format(args.lr))
    print('begin training')
    # training
    for epoch in range(args.epochs):
        for i, (input, affinity) in enumerate(data_generator):
            # use cuda
            # input model
            input_ids, attention_mask = tokenizer.convert_token_to_ids(input)
            pred_affinity = model(input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda())
            loss = loss_fct(pred_affinity, affinity.cuda().unsqueeze(-1))
            step += 1
            writer.add_scalar('loss', loss, global_step=step)
            # Update gradient
            opt.zero_grad()
            loss.backward()
            opt.step()

            #                 if (i % 100 == 0):
            print('Training at Epoch ' + str(epoch + 1) + ' step ' + str(step) + ' with loss ' + str(
                loss.cpu().detach().numpy()))
            # save
            if epoch >= 1 and step % save_step == 0:
                save_path = './model/' + args.savedir + '/'
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                torch.save(model.state_dict(), save_path + 'epoch-{}-step-{}-loss-{}.pth'.format(epoch, step, loss))
    print('training  over')
    writer.close()

def test_mol(args, model, dataset, tokenizer):
    data_loder_para = {'batch_size': args.batch_size,
                       'shuffle': False,
                       'num_workers': args.workers,
                       }
    data_generator = DataLoader(dataset, **data_loder_para)

    with torch.no_grad():
        # if torch.cuda.is_available():
        model.load_state_dict(torch.load(args.init), strict=True)
        model.cuda()
        # else:
        #     model.load_state_dict(torch.load(args.init, map_location=torch.device('cpu')), strict=True)
        model.eval()
        if not os.path.exists(args.output):
            os.mkdir(args.output)
        result = args.output + '/' + '{}.txt'.format(args.task)
        print('begin predicting')
        with open(result, 'w') as f:
            for i, (input, affinity) in enumerate(tqdm(data_generator)):
                input_ids, attention_mask = tokenizer.convert_token_to_ids(input)
                # pred_affinity = model(input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda(), output_attentions=True)
                # attention_mat = pred_affinity["attentions"][-1].detach().cpu().numpy()
                # np.save("visualize_attention/attention_mat", attention_mat)
                pred_affinity = model(input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda())
                                    #   , output_attentions=True)

                # pred_affinity = model(input_ids=input, token_type_ids=token_type_ids, attention_mask=input_mask)
                pred_affinity = pred_affinity.cpu().numpy().squeeze(-1)
                for res in pred_affinity:
                    f.write(str(res) + '\n')

    # if args.do_eval:
    #     os.system('python eval.py')





def main(args):
    # load data
    data_file, tokenizer_config = get_task(args.task)
    if args.task in ['train_mol', 'test_mol', "test_er", "test_gpcr", "test_channel", "test_kinase"]:
        dataset = Data_Gen(data_file)
    else:
        dataset = Data_Encoder(data_file, tokenizer_config)

    # creat model
    print('------------------creat model---------------------------')
    config = BertConfig.from_pretrained(args.config)
    model = BertAffinityModel(config)
    tokenizer = Tokenizer(tokenizer_config)


    print('model name : BertAffinity')
    print('task name : {}'.format(args.task))

    if args.task in ['train_mol']:
        train_mol(args, model, dataset, tokenizer, pre_train=args.pre_train)
        # train(args, model, dataset, tokenizer)

    elif args.task in ['test_mol', "test_er", "test_gpcr", "test_channel", "test_kinase"]:
        test_mol(args, model, dataset, tokenizer)

    elif args.task in ['train', 'train_z_1', 'train_z_10', 'train_z_100']:
        train(args, model, dataset, tokenizer, pre_train=args.pre_train)
        
    elif args.task in ['test', 'test_ori_er', 'test_ori_gpcr', 'test_ori_channel', 'test_ori_kinase']:
        test(args, model, dataset, tokenizer)



if __name__ == '__main__':
    # get parameter
    parser = ArgumentParser(description='BertAffinity')
    parser.add_argument('-batch_size', default=8, type=int,
                        metavar='N',
                        help='mini-batch size (default: 16), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 0)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--task', default='train', type=str, metavar='TASK',
                        help='Task name. Could be train, test, channel, ER, GPCR, kinase or else.')
    parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--config', default='./config/config.json', type=str, help='model config file path')
    # parser.add_argument('--log', default='training_log', type=str, help='training log')
    parser.add_argument('--savedir', default='train', type=str, help='log and model save path')
    # parser.add_argument('--device', default='0', type=str, help='name of GPU')
    parser.add_argument('--init', default='model', type=str, help='init checkpoint')
    parser.add_argument('--output', default='predict', type=str, help='result save path')
    # parser.add_argument('--shuffle', default=True, type=str, help='shuffle data')
    # parser.add_argument('--do_eval', default=False, type=bool, help='do eval')
    parser.add_argument('--pre_train', default=False, type=bool, help='use pre-train')

    args = parser.parse_args()

    # local test
    # os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    # args.task = 'train'
    # args.epochs = 30
    # args.lr = 1e-5
    # args.config = './config/config_layer_6.json'
    # args.savedir = 'train_ori_1217'



    # args.task = 'train_mol'
    # args.savedir = 'without-pre-train-layer-6-1021'
    # # # args.savedir = 'train'
    # args.epochs = 30
    # args.lr = 1e-5
    # args.config = './config/config_layer_6_mol.json'
    # args.pre_train = False
    # args.init = './model/mask-LM-lr-1e-4-1019/epoch-17-step-593064-loss-0.1007341668009758.pth'




    args.task = 'test_mol'
    # args.task = 'test_er'
    # args.task = 'test_gpcr'
    # args.task = 'test_channel'
    # args.task = 'test_kinase'
    args.init = './model/add_pretrain_1019/epoch-9-step-329480-loss-0.736057146887367.pth'
    # args.init = './model/without-pre-train-layer-6-1021/epoch-29-step-988440-loss-0.19894360158554475.pth'
    # args.output = './predict/without-pre-train-layer-6-1021-s-988440-test'
    # args.output = './predict/without-pre-train-layer-6-1021-s-988440-gpcr'
    # args.output = './predict/without-pre-train-layer-6-1021-s-988440-channel'
    # args.output = './predict/without-pre-train-layer-6-1021-s-988440-kinase'
    # args.output = './predict/without-pre-train-layer-6-1021-s-988440-er'
    # args.output = './predict/add_pretrain_1019-s-329480-er'
    # args.output = './predict/add_pretrain_1019-s-329480-gpcr'
    # args.output = './predict/add_pretrain_1019-s-329480-channel'
    # args.output = './predict/add_pretrain_1019-s-329480-kinase'
    args.config = './config/config_layer_6_mol.json'
    args.output = "./predict/test"
    # args.batch_size = 1
    
    
    # test ori
    # args.task = 'test'
    # args.task = 'test_ori_er'
    # args.task = 'test_ori_gpcr'
    # args.task = 'test_ori_channel'
    # args.task = 'test_ori_kinase'
    
    # args.init = 'model/train_ori_1217/epoch-8-step-296532-loss-0.5783637166023254.pth'
    # args.config = './config/config_layer_6.json'
    # args.output = './predict/train_ori_1217-s-296532'
    main(args)

