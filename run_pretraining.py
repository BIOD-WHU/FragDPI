from argparse import ArgumentParser
from dataset import Data_Encoder, get_task, Data_Encoder_mol, Data_Encoder_LM, Tokenizer, Data_Provide
import torch
from torch.utils.data import DataLoader
from configuration_bert import BertConfig
from modeling_bert import BertAffinityModel, BertAffinityModel_MaskLM
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
# torch.set_default_tensor_type(torch.DoubleTensor)
from sklearn.metrics import accuracy_score


def train(args, model, dataset, tokenizer):
    data_loder_para = {'batch_size': args.batch_size,
                       'shuffle': True,
                       'num_workers': args.workers,
                       }
    data_generator = DataLoader(dataset, **data_loder_para)


    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # loss_fct = torch.nn.MSELoss()
    loss_fct = torch.nn.CrossEntropyLoss()
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
        for i, (seq, seq_mask, affinity) in enumerate(data_generator):
            input_random_mask, attention_mask = tokenizer.convert_token_to_ids(seq_mask)
            label, _ = tokenizer.convert_token_to_ids(seq)
            # assert input_random_mask.size() == label.size(), "{}".format(seq_mask)
            logits = model(input_ids=input_random_mask.cuda(), attention_mask=attention_mask.cuda())

            posi = torch.where(input_random_mask == 1)
            pred_logits = logits[posi]
            target = label[posi]
            loss = loss_fct(pred_logits, target.cuda())
            step += 1
            writer.add_scalar('loss', loss, global_step=step)
            # Update gradient
            opt.zero_grad()
            loss.backward()
            opt.step()

            print('Training at Epoch ' + str(epoch + 1) + ' step ' + str(step) + ' with loss ' + str(loss.cpu().detach().numpy()))
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


    # all_pre = []
    # all_label = []
    all_loss = []
    loss_fct = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        model.load_state_dict(torch.load(args.init), strict=True)
        model.eval()
        if not os.path.exists(args.output):
            os.mkdir(args.output)
        result = args.output + '/' + '{}.txt'.format(args.task)
        print('begin predicting')
        with open(result, 'w') as f:
            for i, (seq, seq_mask, affinity) in enumerate(tqdm(data_generator)):
                model.cuda()
                input_random_mask, attention_mask = tokenizer.convert_token_to_ids(seq_mask)
                label, _ = tokenizer.convert_token_to_ids(seq)
                logits = model(input_ids=input_random_mask.cuda(), attention_mask=attention_mask.cuda())
                posi = torch.where(input_random_mask == 1)
                target = label[posi]
                pred_logits = logits[posi]
                pred_p = F.softmax(pred_logits, dim=-1)
                # loss = loss_fct(pred_logits, target.cuda()).detach().cpu().numpy()
                p = pred_p[torch.arange(target.size(0)), target].detach().cpu().numpy().tolist()
                all_loss += p
                # pre = F.softmax(pred_logits, dim=-1).detach().cpu().numpy()
                # pre_id = np.argmax(pre, axis=-1).tolist()
                # all_pre += pre_id
                # all_label += target.tolist()
                # for res in pre:
                #     pre_id = np.argmax(res)
                #     f.write(str(pre_id) + '\n')
            cross_entropy = np.mean(all_loss)
            f.write(str(cross_entropy) + '\n')
            print(cross_entropy)
    # if args.do_eval:
    #     os.system('python eval.py')


def main(args):
    # load data
    data_file, data_mask, tokenizer_config = get_task(args.task)
    # dataset = Data_Encoder(data_file, tokenizer_config)
    dataset = Data_Provide(data_file, data_mask)
    tokenizer = Tokenizer(tokenizer_config)
    # creat model
    print('------------------creat model---------------------------')
    config = BertConfig.from_pretrained(args.config)
    model = BertAffinityModel_MaskLM(config)

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = torch.nn.DataParallel(model, dim=0)

    print('model name : BertAffinity')
    print('task name : {}'.format(args.task))

    if args.task in ['pre-train']:
        train(args, model, dataset, tokenizer)

    elif args.task in ["test-pre-train"]:
        test(args, model, dataset, tokenizer)




if __name__ == '__main__':
    # get parameter
    parser = ArgumentParser(description='BertAffinity')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        metavar='N',
                        help='mini-batch size (default: 8), this is the total '
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
    parser.add_argument('--do_eval', default=False, type=bool, help='do eval')

    args = parser.parse_args()

    # local test
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    # args.task = 'pre-train'
    # args.savedir = 'mask-LM-layer-6-dobule-1020'
    # # args.savedir = 'train'
    # args.epochs = 30
    # args.lr = 1e-4
    # args.config = './config/config_layer_6_mol.json'



    args.task = 'test-pre-train'
    args.init = './model/add_pretrain_1019/epoch-9-step-329480-loss-0.736057146887367.pth'
    args.output = './predict/test-pre-train'
    args.config = './config/config_layer_6_mol.json'
    main(args)

