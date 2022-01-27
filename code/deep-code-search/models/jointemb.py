import os
import sys

import torch
import torch.nn as nn
import torch.nn.init as weight_init
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)
parserPath = os.path.abspath("..")
sys.path.insert(0, parserPath)  # add parent folder to path so as to import common modules
from modules import SeqEncoder, BOWEncoder

class JointEmbeder(nn.Module):
    """
    References on sentence pair matching models:
        https://arxiv.org/pdf/1508.01585.pdf
        https://arxiv.org/pdf/1908.10084.pdf
    similarity scale classification for sentence pairs:
        https://arxiv.org/pdf/1503.00075.pdf
    """
    def __init__(self, config):
        super(JointEmbeder, self).__init__()
        self.conf = config
        self.margin = config['margin']

        self.name_encoder = SeqEncoder(config['n_words'], config['emb_size'], config['lstm_dims'])
        self.api_encoder = SeqEncoder(config['n_words'], config['emb_size'], config['lstm_dims'])
        self.tok_encoder = BOWEncoder(config['n_words'], config['emb_size'], config['n_hidden'])
        self.desc_encoder = SeqEncoder(config['n_words'], config['emb_size'], config['lstm_dims'])

        # self.fuse1 = nn.Linear(config['emb_size']+4*config['lstm_dims'], config['n_hidden'])
        # self.fuse2 = nn.Sequential(
        #     nn.Linear(config['emb_size']+4*config['lstm_dims'], config['n_hidden']),
        #     nn.BatchNorm1d(config['n_hidden'], eps=1e-05, momentum=0.1),
        #     nn.ReLU(),
        #     nn.Linear(config['n_hidden'], config['n_hidden']),
        # )
        self.w_name = nn.Linear(2*config['lstm_dims'], config['n_hidden'])
        self.w_api = nn.Linear(2*config['lstm_dims'], config['n_hidden'])
        self.w_tok = nn.Linear(config['emb_size'], config['n_hidden'])
        self.w_desc = nn.Linear(2*config['lstm_dims'], config['n_hidden'])
        self.fuse3 = nn.Linear(config['n_hidden'], config['n_hidden'])

        self.init_weights()

    def init_weights(self):
        # initialize linear wegiht
        for m in [self.w_name, self.w_api, self.tok, self.fuse3]:
            m.weight.data.uniform_(-0.1, 0.1)  # nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.)

    def code_