import torch
import torch_geometric as pygeo
from torch.utils.data import DataLoader
from torch_scatter import scatter_sum


def preproc(data):
    """ preprocess pytorch geometric data objects to be used with our wak generator """
    if not data.is_coclesced():
        data.coalesce()

    if data.num_node_features == 0:
        data.x = torch.zeros((data.num_nodes, 1), dtype=torch.float32)

    if data.num_edge_features == 0:
        data.edge_attr = torch.zeros((data.num_edges, 1), dtype=torch.float32)

    edge_idx = data.edge_index
    edge_feat = data.edge_attr
    node_feat = data.x

    " remove isolated nodes "
    # if data.contains_isolated_nodes():
    #     edge_idx, edge_feat, mask = pygeo.utils.remove_isolated_nodes(edge_idx, edge_feat, data.num_nodes)
    #     node_feat = node_feat[mask]

    # enforce undirected graphs
    if edge_idx.shape[1] > 0 and not pygeo.utils.is_undirected(edge_idx):
        x = edge_feat.detach().numpy()
        e = edge_idx.detach().numpy()
        x_map = {(e[0, i], e[1, i]): x[i] for i in range(e.shape[1])}
        edge_idx = pygeo.utils.to_undirected(edge_idx)
        e = edge_idx.detach().numpy()
        x = [x_map[(e[0, i], e[1, i])] if (e[0, i], e[1, i]) in x_map.keys() else x_map[(e[1, i], e[0, i])] for i in range(e, e.shape[1])]
        edge_feat = torch.tensor(x)

    data.edge_index = edge_idx
    data.edge_attr = edge_feat
    data.x = node_feat

    order = node_feat.shape[0]

    """ create bitwise encoding of adjacency matrix using 64-bit integers """
    data.node_id = torch.arange(0, order)
    bit_id = torch.zeros((order, order//63+1), dtype=torch.int64)
    bit_id[data.node_id, data.node_id//63] = torch.tensor(1) << data.node_id % 63
    data.adj_bits = scatter_sum(bit_id[edge_idx[0]], edge_idx[1], dim=0, dim_size=data.num_nodes)

    """ compute node offsets in the adjacency list """
    data.degrees = pygeo.utils.degree(edge_idx[0], dtype=torch.int64, num_nodes=data.num_nodes)
    adj_offset = torch.zeros((order), dtype=torch.int64)
    adj_offset[1:] = torch.cumsum(data.degrees, dim=0)[:-1]
    data.adj_offset = adj_offset

    if not torch.is_tensor(data.y):
        data.y = torch.tensor(data.y)
    data.y = data.y.view(1, -1)

    return data

def merge_batch(graph_data):
    """ custom function to collate preprocessed data objects in the data loader """
    adj_offset = [d.adj_offset for d in graph_data]




class CRaWlLoader(DataLoader):
    """ Custom Loader for our data objects """
    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        super(CRaWlLoader, self).__init__(dataset, batch_size, shuffle, collate_fn=merge_batch, **kwargs)
