from argparse import ArgumentParser
from dataset import Data_Encoder, get_task, Data_Encoder_mol, Data_Gen, Tokenizer
import torch
from torch.utils.data import DataLoader
from configuration_bert import BertConfig
from modeling_bert import BertAffinityModel
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm


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
            for i, input in enumerate(tqdm(data_generator)):
                input_ids, attention_mask = tokenizer.convert_token_to_ids(input)
                pred_affinity = model(input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda())
                # pred_affinity = model(input_ids=input, token_type_ids=token_type_ids, attention_mask=input_mask)
                pred_affinity = pred_affinity.cpu().numpy().squeeze(-1)
                for res in pred_affinity:
                    f.write(str(res) + '\n')

    # if args.do_eval:
    #     os.system('python eval.py')
    
    


def main(args):
    # load data
    data_file, tokenizer_config = get_task(args.task)
    # if args.task in ['train_mol', 'test_mol', "test_er", "test_gpcr", "test_channel", "test_kinase"]:
        # dataset = Data_Gen(data_file)
    # else:
    #     dataset = Data_Encoder(data_file, tokenizer_config)
    dataset = Data_Gen(data_file)
    # creat model
    print('------------------creat model---------------------------')
    config = BertConfig.from_pretrained(args.config)
    model = BertAffinityModel(config)
    tokenizer = Tokenizer(tokenizer_config)


    print('model name : BertAffinity')
    print('task name : {}'.format(args.task))

    test_mol(args, model, dataset, tokenizer)
    # if args.task in ['train_mol']:
    #     train_mol(args, model, dataset, tokenizer, pre_train=args.pre_train)
    #     # train(args, model, dataset, tokenizer)

    # elif args.task in ['test_mol', "test_er", "test_gpcr", "test_channel", "test_kinase"]:
    #     test_mol(args, model, dataset, tokenizer)

    # elif args.task in ['train', 'train_z_1', 'train_z_10', 'train_z_100']:
    #     train(args, model, dataset, tokenizer, pre_train=args.pre_train)
        
    # elif args.task in ['test', 'test_ori_er', 'test_ori_gpcr', 'test_ori_channel', 'test_ori_kinase']:
    #     test(args, model, dataset, tokenizer)
    
    

if __name__ == '__main__':
    parser = ArgumentParser(description='BertAffinity')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"







    args.task = 'case_study'

    args.init = './model/add_pretrain_1019/epoch-9-step-329480-loss-0.736057146887367.pth'

    args.config = './config/config_layer_6_mol.json'
    
    args.output = 'case_study/output'
    main(args)