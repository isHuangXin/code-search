import logging
import os
import random
import sys
import argparse
import numpy as np
import torch.backends.cudnn
import configs


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

from datetime import datetime
from tensorboardX import SummaryWriter


try:
    import nsml
    from nsml import DATASET_PATH, IS_ON_NSML, SESSION_NAME
except:
    IS_ON_NSML = False

def train(args):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M')
    print(timestamp)
    # make ouput directory if it doesn't already exist
    # os.makedirs(f'./output/{args.model}/{args.dataset}/{timestamp}/models', exist_ok=True)
    # os.makedirs(f'./output/{args.model}/{args.dataset}/{timestamp}/tmp_results', exist_ok=True)
    os.makedirs(f'./output/{args.model}/{args.dataset}/models', exist_ok=True)
    os.makedirs(f'./output/{args.model}/{args.dataset}/tmp_results', exist_ok=True)

    " create file handler with logs even debug messages "
    " add the handlers to the logger "
    # fh = logging.FileHandler(f'./output/{args.model}/{args.dataset}/{timestamp}/logs.txt')
    # logger.addHandler(fh)
    # tb_writber = SummaryWriter(f'./output/{args.model}/{args.dataset}/{timestamp}/logs/') if args.visual else None
    fh = logging.FileHandler(f'./output/{args.model}/{args.dataset}/logs.txt')
    logger.addHandler(fh)
    tb_writber = SummaryWriter(f'./output/{args.model}/{args.dataset}/logs/') if args.visual else None

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    config = getattr(configs, 'config_'+args.model)()
    if args.automl:
        config.update(vars(args))
    print(config)

    " Load data "
    data_path = DATASET_PATH + '/train/' if IS_ON_NSML else args.data_path+args.dataset+'/'
    print(data_path)
    train_set = eval(config['dataset_name'])(data_path, config['train_name'], config['name_len'],
                                              config['train_api'], config['api_len'],
                                              config['train_tokens'], config['tokens_len'],
                                              config['train_desc'], config['desc_len'])
    valid_set = eval(config['dataset_name'])(data_path,
                                             config['valid_name'], config['name_len'],
                                             config['valid_api'], config['api_len'],
                                             config['valid_tokens'], config['tokens_len'],
                                             config['valid_desc'], config['desc_len'])
    data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=config['batch_size'],
                                              shuffle=True, drop_last=True, num_workers=1)


def parser_args():
    parser = argparse.ArgumentParser("Train and Validate The Code Search (Embedding) Model")
    parser.add_argument('--data_path', type=str, default='/home/wanyao/huangxin/data/deep_code_search/', help='location of the data corpus')
    parser.add_argument('--model', type=str, default='JointEmbeder', help='model name: JointEmbeder, SelfAttnModel')
    parser.add_argument('--dataset', type=str, default='github', help='name of dataset.java, python')
    parser.add_argument('--reload_from', type=int, default=-1, help='epoch to reload from')

    parser.add_argument('-g', '--gpu_id', type=int, default=0, help='GPU ID')
    # 若触发 test_action，则为 True， 否则为 False
    parser.add_argument('-v', '--visual', action='store_true', default=False, help='Visualize training status in tensorboard')
    parser.add_argument('--automl', action='store_true', default=False, help='use automl')

    # Training Arguments
    parser.add_argument('--log_every', type=int, default=100, help='interval to log autoencoder training results')
    parser.add_argument('--valid_every', type=int, default=10000, help='interval to validation')
    parser.add_argument('--save_every', type=int, default=50000, help='interval to evaluation to concrete results')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')


    return parser.parse_args()



if __name__ == '__main__':
    args = parser_args()
    """ speed up training by using cudnn """
    torch.backends.cudnn.benchmark = True
    """ fix the random seed in cudnn """
    torch.backends.cudnn.deterministic = True
    train(args)
