import sys

import numpy as np
import torch
import torch.utils.data as data
import tables


class CodeSearchDataset(data.Dataset):
    """
    Dataset that has only positive samples.
    """
    def __init__(self, data_dir, f_name, max_name_len, f_api, max_api_len,
                       f_tokens, max_tok_len, f_desc=None, max_desc_len=None):
        self.max_name_len = max_name_len
        self.max_api_len = max_api_len
        self.max_token_len = max_tok_len
        self.max_desc_len = max_desc_len
        # 1. Initialize file path or list of file names.
        """ read training data(list of int arrays) from a hdf5 file"""
        self.training = False
        print("loading data ...")
        table_name = tables.open_file(data_dir + f_name)
        self.names = table_name.get_node('/phrases')[:].astype(np.long)
        self.idx_names = table_name.get_node('/indices')[:]
        table_api = tables.open_file(data_dir + f_api)
        self.apis = table_api.get_node('/phrases')[:].astype(np.long)
        self.idx_apis = table_api.get_node('/indices')[:]
        table_tokens = tables.open_file(data_dir + f_tokens)
        self.tokens = table_tokens.get_node('/phrases')[:].astype(np.long)
        self.idx_tokens = table_tokens.get_node('/indices')[:]
        if f_desc is not None:
            self.training = True
            table_desc = tables.open_file(data_dir + f_desc)
            self.descs = table_desc.get_node('/phrases')[:].astype(np.long)
            self.idx_descs = table_desc.get_node('/indices')[:]

        assert self.idx_names.shape[0] == self.idx_apis.shape[0]
        assert self.idx_apis.shape[0] == self.idx_tokens.shape[0]
        if f_desc is not None:
            assert self.idx_names.shape[0] == self.idx_descs.shape[0]
        self.data_len = self.idx_names.shape[0]
        print("{} entries".format(self.data_len))


