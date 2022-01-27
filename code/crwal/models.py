import torch
from torch.nn import Module, Linear, Sequential, Conv1d, BatchNorm1d, ReLU


class ConvModule(Module):
    def __init__(self, conv_dim, node_dim_in, edge_dim_in, w_feat_dim, dim_out, kernel_size):
        """
        :param conv_dim: hidden dimension of the convolutions
        :param node_dim_in: input dimension of the node features
        :param edge_dim_in: input dimension of the edge features
        :param w_feat_dim: dimension of the structural encodings of the walk feature tensor (A and I)
        :param dim_out: dimension of the updated latent node embedding
        :param kernel_size: kernel size of the convolutions (usually chosen as s+1)
        """
        super(ConvModule, self).__init__()
        self.node_dim_in = node_dim_in
        self.edge_dim_in = edge_dim_in
        self.kernel_size = kernel_size

        # pool into center node
        self.node_dim_in = node_dim_in
        self.edge_dim_in = edge_dim_in
        self.kernel_size = kernel_size

        # pool into center node
        self.pool_node = kernel_size // 2

        # rescale for  residual connection
        self.node_rescale = Linear(node_dim_in, dim_out, bias=False) if node_dim_in != dim_out else Identity()

        # lost nodes due to lack of padding
        self.border = kernel_size - 1

        self.convs = Sequential(
            Conv1d(node_dim_in + edge_dim_in + w_feat_dim, conv_dim, 1, padding=0, bias=False),
            Conv1d(conv_dim, conv_dim, kernel_size, groups=conv_dim, padding=0, bias=False),
            BatchNorm1d(conv_dim),
            ReLU(),
            Conv1d(conv_dim, conv_dim, 1, padding=0, bias=False),
            ReLU()
        )
        self.node_out = Sequential(Linear(conv_dim, 2*dim_out, bias=False),
                                   BatchNorm1d(2*dim_out),
                                   ReLU(),
                                   Linear(2*dim_out, dim_out, bias=False))

    def forward(self, data):
        walk_nodes = data.walk_nodes

        # build walk feature tensor
        walk_node_h = data.h[walk_nodes].transpose(2, 1)
        if 'walk_edge_h' not in data:
            padding = torch.zeros((walk_node_h.shape[0], self.edge_dim_in, 1), dtype=torch.float32, device=walk_node_h.device)
            data.walk_edge_h = torch.cat([padding, data.edge_h[data.walk_edges].transpose(2, 1)], dim=2)
        if 'walk_x' in data:
            x = torch.cat([walk_node_h, data.walk_edge_h, data.walk_x], dim=1)
        else:
            x = torch.cat([walk_node_h, data.walk_edge_h], dim=1)

        # apply the cnn
        y = self.convs(x)

        # pool in walklet embeddings into nodes
        flatt_dim = y.shape[0] * y.shape[2]
        y_flatt = y.transpose(2, 1).reshape(flatt_dim, -1)

        # get center indices
        if 'walk_nodes_flatt' not in data:
            data.walk_nodes_flatt = walk_nodes[:, self.pool_node:-(self.kernel_size - 1 - self.pool_node)].reshape(-1)

        # pool graphlet embeddings into nodes
        p_node = scatter_mean(y_flatt, data.walk_nodes_flatt, dim=0, dim_size=data.num_nodes)

        # rescale for the residual connection
        data.h = self.node_rescale(data.h)
        data.h += self.node_out(p_node)

        return data



class CRaWl(Module):
    def __init__(self, model_dir, config, node_feat_dim, edge_feat_dim, out_dim, loss, node_feat_enc=Node, edge_feat_enc=None):
        """
        :param model_dir: directory to store model in
        :param config: python dict that specifies the configuration of the model
        :param node_feat_dim: dimension of the node features
        :param edge_feat_dim: dimension of the edge features
        :param out_dim: output dimension
        :param loss: torch.nn loss object used for training
        :param node_feat_enc: optional initial embedding of node features
        :param edge_feat_enc: optional initial embedding of edge features
        """
        super(CRaWl, self).__init__()
        self.model_dir = model_dir
        self.config = config
        self.out_dim = out_dim
        self.node_feat_enc = node_feat_enc
        self.edge_feat_enc = edge_feat_enc

        self.layers = config['layers']
        self.hidden = config['hidden_dim']
        self.kernel_size = config['kernel_size']
        self.dropout = config['dropout']

        self.pool = config['pool'] if 'pool' in config.keys() else 'mean'
        self.vn = config['vn'] if 'vn' in config.keys() else False

        self.walker = Walker(config)














