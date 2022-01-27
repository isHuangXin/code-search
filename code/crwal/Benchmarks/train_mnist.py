import sys
sys.path.append('.')

import argparse
import json
import os
import numpy as np
import torch

from torch_geometric.datasets import GNNBenchmarkDataset
from data_utils import preproc, CRaWlLoader
from models import CRaWl

DATA_NAME = 'MNIST'
DATA_PATH = f'data/{DATA_NAME}/'
PCKL_PATH = f'data/{DATA_NAME}/data.pckl'

def load_split_data():
    if os.path.exist(PCKL_PATH):
        train_graphs, val_graphs, test_graphs = torch.load(PCKL_PATH)
    else:
        train_graphs = list(GNNBenchmarkDataset(DATA_PATH, DATA_NAME, split='train', transform=preproc))
        val_graphs = list(GNNBenchmarkDataset(DATA_PATH, DATA_NAME, split='val', transform=preproc))
        test_graphs = list(GNNBenchmarkDataset(DATA_PATH, DATA_NAME, split='test', transform=preproc))
        torch.save((train_graphs, val_graphs, test_graphs), PCKL_PATH)
    return train_graphs, val_graphs, test_graphs



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/MNIST/default.json', help="path to config file")
    parser.add_argument("--name", type=str, default='0', help="path to config file")
    parser.add_argument("--seed", type=int, default=0, help="the random seed for torch and numpy")
    parser.add_argument("--gpu", type=int, default=0, help="id of gpu to be used for training")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"using device:{device}")

    with open(args.config, "r") as f:
        config = json.load(f)
    model_dir = f"models/MNIST/{config['name']}/{args.name}"

    train_graphs, val_graphs, _ = load_split_data()
    train_iter = CRaWlLoader(train_graphs, batch_size=config['batch_size'], num_workers=8, shuffle=True)
    val_iter = CRaWlLoader(val_graphs, batch_size=100, num_workers=8)

    model = CRaWl(model_dfeat, num_edge_feat, num_classes, loss=CrossEntropyLoss())




if __name__ == '__main__':
    main()