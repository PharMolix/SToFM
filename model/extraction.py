import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import scanpy as sc, pandas as pd
from sklearn.decomposition import PCA
from datasets import load_from_disk
from .utils import CellEncoderCollator
import datetime, pytz
import rapids_singlecell as rsc, cupy as cp
import torch.distributed as dist
from scipy.spatial.distance import cdist

def gather_graphs(local_rank, world_size, graphs):
    rank2graphs = {}
    for src_rank in range(world_size):
        # if local_rank == 0:
            # print(f"gathering from rank {src_rank}")
        if src_rank == local_rank:
            dist.broadcast(torch.tensor(len(graphs)), src=src_rank)
            for graph in graphs:
                dist.broadcast(torch.tensor(graph['spatial'].shape[0]), src=src_rank)
                dist.broadcast(graph['spatial'], src=src_rank)
                dist.broadcast(graph['token_embeddings'], src=src_rank)
                dist.broadcast(graph['indices'], src=src_rank)
                dist.broadcast(graph['attn_bias'], src=src_rank)
                dist.broadcast(graph['token_types'], src=src_rank)
            rank2graphs[src_rank] = graphs
        else:
            new_graphs = []
            len_graphs = torch.tensor(0)
            dist.broadcast(len_graphs, src=src_rank)
            for i in range(len_graphs.item()):
                node_num = torch.tensor(0)
                dist.broadcast(node_num, src=src_rank)
                node_num = node_num.item()
                spatial = torch.zeros((node_num, 2), dtype=torch.float32)
                dist.broadcast(spatial, src=src_rank)
                token_embeddings = torch.zeros((node_num, graphs[0]['token_embeddings'].shape[1]), dtype=torch.float32)
                dist.broadcast(token_embeddings, src=src_rank)
                indices = torch.zeros((node_num), dtype=torch.int32)
                dist.broadcast(indices, src=src_rank)
                attn_bias = torch.zeros((node_num, node_num), dtype=torch.float32)
                dist.broadcast(attn_bias, src=src_rank)
                token_types = torch.zeros((node_num), dtype=torch.int32)
                dist.broadcast(token_types, src=src_rank)
                new_graphs.append({'spatial': spatial, 'token_embeddings': token_embeddings, 
                                    'indices': indices, 'attn_bias': attn_bias, 'token_types': token_types})
            rank2graphs[src_rank] = new_graphs
    return rank2graphs


def encode_cell(model, model_input_path, emb_path, save=True, batch_size=1, add_cls=True):
    if model is None or model_input_path is None:
        raise ValueError("model or model_input_path is missing")
    model_input = load_from_disk(model_input_path)#, keep_in_memory=True)
    model_input = model_input.remove_columns(set(model_input.column_names) - set(['input_ids']))
    collator = CellEncoderCollator(add_cls=add_cls)
    dataloader = DataLoader(model_input, batch_size=batch_size, collate_fn=collator, shuffle=False)
    cell_embs = []
    with torch.no_grad():
        for d in dataloader:
            out = model(d['input_ids'].to(model.device), d['attention_mask'].to(model.device))
            cellemb = out.pooler_output if model.pooler is not None else F.normalize(out.last_hidden_state[:, 0, :], dim=-1)
            cell_embs.append(cellemb)
        cell_embs = torch.cat(cell_embs, dim=0).cpu()
    cell_embs_np = cell_embs.numpy()
    if save and emb_path is not None:
        np.save(emb_path, cell_embs_np)
        print(f"Save cell embeddings to {emb_path}")
    return cell_embs_np

def get_hypernodes(nodes, features, alpha=0.2, device=-1, leiden_res=1.0): # nodes: [n, 2], features: [n, d], alpha: a * feature_pca + (1 - a) * spatial
    # cluster
    norm_nodes = nodes / np.sqrt(np.max(np.sum(nodes ** 2, axis=1)))
    pca_features = PCA(n_components=2).fit_transform(features)
    norm_pca_features = pca_features / np.sqrt(np.max(np.sum(pca_features ** 2, axis=1)))
    cluster_features = alpha * norm_nodes + (1 - alpha) * norm_pca_features
    nodes_an = sc.AnnData(X=cluster_features)#, dtype=cluster_features.dtype)
    if device >= 0:
        try:
            with cp.cuda.Device(device):
                rsc.get.anndata_to_GPU(nodes_an)
                rsc.pp.neighbors(nodes_an, n_neighbors=10, use_rep='X')
                resolution = 1.0
                while(True):
                    rsc.tl.leiden(nodes_an, resolution=resolution)
                    if len(np.unique(nodes_an.obs['leiden'])) < 200:
                        break
                    resolution /= 2
                rsc.get.anndata_to_CPU(nodes_an)
        except:
            rsc.get.anndata_to_CPU(nodes_an)
            sc.pp.neighbors(nodes_an, n_neighbors=10, use_rep='X')
            # sc.tl.leiden(nodes_an, flavor='igraph')
            resolution = 1.0
            while(True):
                sc.tl.leiden(nodes_an, resolution=resolution)
                if len(np.unique(nodes_an.obs['leiden'])) < 200:
                    break
                resolution /= 2
    else:
        sc.pp.neighbors(nodes_an, n_neighbors=10, use_rep='X')
    # some bugs in rsc.tl.leiden? use scanpy.tl.leiden
    sc.tl.leiden(nodes_an, flavor='igraph', resolution=leiden_res)
    cluster = nodes_an.obs['leiden'].to_numpy().astype(int)
    # for each cluster, get the hypernode
    hypernodes = []
    hypernode_features = []
    for i in np.unique(cluster):
        cluster_nodes = nodes[cluster == i]
        cluster_features = features[cluster == i]
        hypernode = np.mean(cluster_nodes, axis=0)
        hypernode_feature = np.mean(cluster_features, axis=0)
        hypernodes.append(hypernode)
        hypernode_features.append(hypernode_feature)
    return np.array(hypernodes), np.array(hypernode_features) # [cluster, 2], [cluster, d]

def split_graph(nodes, features, num=1000): # nodes: [n, 2], features: [n, d]
    node_num = nodes.shape[0]
    if node_num <= num:
        return [nodes], [features], [np.arange(node_num)]
    max_x = np.max(nodes[:, 0])
    min_x = np.min(nodes[:, 0])
    max_y = np.max(nodes[:, 1])
    min_y = np.min(nodes[:, 1])
    x_len = max_x - min_x
    y_len = max_y - min_y
    # split成正方形
    split_num = node_num // num + 1
    split_len = np.sqrt(x_len * y_len / split_num)
    split_x = int(x_len / split_len) + 1
    split_y = int(y_len // split_len) + 1
    split_indices = []
    split_nodes = []
    split_features = []
    for i in range(split_x):
        for j in range(split_y):
            x_min = min_x + i * split_len
            x_max = min_x + (i + 1) * split_len
            y_min = min_y + j * split_len
            y_max = min_y + (j + 1) * split_len
            mask = (nodes[:, 0] >= x_min) & (nodes[:, 0] < x_max) & (nodes[:, 1] >= y_min) & (nodes[:, 1] < y_max)
            indice = np.where(mask)[0]
            # 确保每个split至少有一个node，且node数量不超过num * 2
            if len(indice) == 0:
                continue
            elif len(indice) // num >= 2:
                indices = np.array_split(indice, len(indice) // num)
                for new_indice in indices:
                    split_indices.append(new_indice)
                    split_nodes.append(nodes[new_indice])
                    split_features.append(features[new_indice])
            else:
                split_indices.append(indice)
                split_nodes.append(nodes[mask])
                split_features.append(features[mask])
    return split_nodes, split_features, split_indices

def preprocess(nodes, features, num=1000, device=-1, leiden_res=1.0, alpha=0.2):
    split_nodes, split_features, split_indices = split_graph(nodes, features, num)
    hypernodes, hypernode_features = get_hypernodes(nodes, features, device=device, leiden_res=leiden_res, alpha=alpha)
    return hypernodes, hypernode_features, split_nodes, split_features, split_indices

def get_input(nodes, features, indices, token_types, device=-1, hyper_type=2): 
        # nodes: [num_nodes, 2], features: [num_nodes, dim], indices: [num_nodes], token_types: [num_nodes]
        num_nodes, hidden_size = features.shape 
        nodes_an = sc.AnnData(X=nodes)#, dtype=nodes.dtype)
        # here should take a bigger n_neighbors
        # hypernodes should have be neighbors with all nodes
        n_neighbors = min(50, num_nodes - 1)
        if device >= 0:
            try:
                with cp.cuda.Device(device):
                    # print(f"use rsc to get connect on GPU {device}")
                    rsc.get.anndata_to_GPU(nodes_an)
                    rsc.pp.neighbors(nodes_an, n_neighbors=n_neighbors, use_rep='X')
                    rsc.get.anndata_to_CPU(nodes_an)
            except Exception as e:
                print(f"rank {device}, rsc bug: {e}")
                np.save("debug/nodes.npy", nodes)
                np.save("debug/token_types.npy", token_types)
                rsc.get.anndata_to_CPU(nodes_an)
                sc.pp.neighbors(nodes_an, n_neighbors=n_neighbors, use_rep='X')
        else:
            sc.pp.neighbors(nodes_an, n_neighbors=n_neighbors, use_rep='X')

        # calculate distances of hypernodes
        hyper_nodes = nodes[token_types == 2]
        hyper_distances = cdist(hyper_nodes, nodes, metric='euclidean')
        # print(f"hyper_distances shape: {hyper_distances.shape}")

        attn_bias = np.zeros([num_nodes, num_nodes], dtype=np.single)  # with graph token
        attn_bias[1:, 1:] = nodes_an.obsp['distances'].toarray()[1:, 1:]
        attn_bias[token_types == 2, 1:] = hyper_distances[:, 1:]
        attn_bias[1:, token_types == 2] = hyper_distances[:, 1:].T
        attn_bias[attn_bias == 0] = 100.
        attn_bias[:, 0] = 0.
        attn_bias[0, :] = 0.
        # 对称
        attn_bias[attn_bias == 100.] += (attn_bias.T[attn_bias == 100.] - 100) * (attn_bias.T[attn_bias == 100.] != 100)
        for i in range(num_nodes):
            attn_bias[i, i] = 0.
        attn_bias = torch.from_numpy(attn_bias).to(torch.float32)

        spatial = torch.from_numpy(nodes).to(torch.float32)
        token_embeddings = torch.from_numpy(features).to(torch.float32)
        indices = torch.from_numpy(indices).to(torch.int32)
        token_types = token_types.to(torch.int32)

        input = {'spatial': spatial, 'token_embeddings': token_embeddings, 'indices': indices,
                 'attn_bias': attn_bias, 'token_types': token_types}
        return input

def load_data(data_path, spatial_path=None, emb_path=None, 
              new_emb=False, model=None, model_input_path=None,
              norm_type=0, cls_type=1, hyper_type=2, pad_type=3,
              device=-1, filter=True, split_num=1000, leiden_res=1.0, 
              alpha=0.2, **kwargs):
    if data_path.endswith(".h5ad"):
        adata = sc.read_h5ad(data_path)
    else:
        adata = sc.read_10x_mtx(data_path)
    adata.var_names_make_unique()

    if 'spatial' not in adata.obsm_keys():
        spatial_key = None
        for k in adata.obsm_keys():
            if "spatial" in k and (adata.obsm[k].shape[1] == 2 or adata.obsm[k].shape[1] == 3):
                spatial_key = k
                break
        if spatial_key is not None:
            adata.obsm["spatial"] = adata.obsm[spatial_key]
        elif spatial_path is not None and spatial_path.endswith(".csv"):
                spatial_df = pd.read_csv(spatial_path, index_col=0)
                spatial_df = pd.merge(adata.obs, spatial_df, left_index=True, right_index=True)[["spatial_1","spatial_2"]]
                adata.obsm["spatial"] = np.array(spatial_df.values)
        else:
            raise ValueError("spatial information is missing")
    if not new_emb:
        if emb_path is None:
            raise ValueError("emb_path is missing")
        else:
            adata.obsm["cell_emb"] = np.load(emb_path)
    else:
        adata.obsm["cell_emb"] = encode_cell(model, model_input_path, emb_path, save=True)

    if filter:
        sc.pp.filter_cells(adata, min_genes=20)
    # adata = adata[adata
    cell_emb = adata.obsm["cell_emb"]
    spatial = adata.obsm["spatial"][:, :2] # only use x, y if z is provided
    del adata

    # normlize spatial
    max_x = np.max(spatial[:, 0])
    min_x = np.min(spatial[:, 0])
    max_y = np.max(spatial[:, 1])
    min_y = np.min(spatial[:, 1])
    x_mid = (max_x + min_x) / 2
    y_mid = (max_y + min_y) / 2
    graph_size = ((max_x - min_x) ** 2 + (max_y - min_y) ** 2) ** 0.5
    spatial = (spatial - [x_mid, y_mid]) / graph_size * 100

    hypernodes, hypernode_features, split_nodes, split_features, split_indices = preprocess(spatial, cell_emb, device=device, 
                                                                                            num=split_num, leiden_res=leiden_res, alpha=alpha)
    inputs = []
    for split_node, split_feature, split_indice in zip(split_nodes, split_features, split_indices):
        split_node = np.concatenate([np.zeros((1, *(hypernodes.shape[1:]))), hypernodes, split_node], axis=0) # cls + [hyper] + [nodes]
        split_feature = np.concatenate([np.zeros((1, *(hypernode_features.shape[1:]))), hypernode_features, split_feature], axis=0) # cls + [hyper] + [nodes]
        split_indice = np.concatenate([np.ones(1 + len(hypernodes), dtype=int) * (-1), split_indice]) # [-1](cls + [hyper]) + [indices]
        
        token_types = torch.ones(split_node.shape[0], dtype=torch.int32) * norm_type # cls + [hyper] + [nodes]
        token_types[0] = cls_type
        token_types[1: len(hypernodes) + 1] = hyper_type
        input = get_input(split_node, split_feature, split_indice, token_types, device=device, hyper_type=hyper_type)
        # print(split_node.shape, split_feature.shape, split_indice.shape, token_types.shape)
        inputs.append(input)
    return inputs

def load_data_debug(data_path, spatial_path=None, emb_path=None, 
              new_emb=False, model=None, model_input_path=None,
              norm_type=0, cls_type=1, hyper_type=2, pad_type=3,
              device=-1):
    time1 = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    if data_path.endswith(".h5ad"):
        adata = sc.read_h5ad(data_path)
    else:
        adata = sc.read_10x_mtx(data_path)
    adata.var_names_make_unique()
    time2 = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    print(f"Device {device}, load data time: {time2 - time1}")

    if 'spatial' not in adata.obsm_keys():
        spatial_key = None
        for k in adata.obsm_keys():
            if "spatial" in k:
                spatial_key = k
                break
        if spatial_key is not None:
            adata.obsm["spatial"] = adata.obsm[spatial_key]
        elif spatial_path is not None and spatial_path.endswith(".csv"):
                spatial_df = pd.read_csv(spatial_path, index_col=0)
                spatial_df = pd.merge(adata.obs, spatial_df, left_index=True, right_index=True)[["spatial_1","spatial_2"]]
                adata.obsm["spatial"] = np.array(spatial_df.values)
        else:
            raise ValueError("spatial information is missing")
    if not new_emb:
        if emb_path is None:
            raise ValueError("emb_path is missing")
        else:
            adata.obsm["cell_emb"] = np.load(emb_path)
    else:
        adata.obsm["cell_emb"] = encode_cell(model, model_input_path, emb_path, save=True)
    time3 = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    print(f"Device {device}, encode cell time: {time3 - time2}")

    sc.pp.filter_cells(adata, min_genes=20)
    cell_emb = adata.obsm["cell_emb"]
    spatial = adata.obsm["spatial"]
    # hypernodes, hypernode_features, split_nodes, split_features, split_indices = preprocess(spatial, cell_emb)

    split_nodes, split_features, split_indices = split_graph(spatial, cell_emb, 1000)
    time4 = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    print(f"Device {device}, split graph time: {time4 - time3}")

    hypernodes, hypernode_features = get_hypernodes(spatial, cell_emb, device=device)
    time5 = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    print(f"Device {device}, get hypernodes time: {time5 - time4}, hypernodes shape: {hypernodes.shape}")


    inputs = []
    for split_node, split_feature, split_indice in zip(split_nodes, split_features, split_indices):
        split_node = np.concatenate([np.zeros((1, *hypernodes.shape[1:])), hypernodes, split_node], axis=0) # cls + [hyper] + [nodes]
        split_feature = np.concatenate([np.zeros((1, *hypernode_features.shape[1:])), hypernode_features, split_feature], axis=0) # cls + [hyper] + [nodes]
        split_indice = np.concatenate([np.ones(1 + len(hypernodes), dtype=int) * (-1), split_indice]) # [-1](cls + [hyper]) + [indices]
        
        token_types = torch.ones(split_node.shape[0], dtype=torch.int32) * norm_type # cls + [hyper] + [nodes]
        token_types[0] = cls_type
        token_types[1: len(hypernodes) + 1] = hyper_type
        input = get_input(split_node, split_feature, split_indice, token_types, device=device)
        # print(split_node.shape, split_feature.shape, split_indice.shape, token_types.shape)
        inputs.append(input)
    time6 = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    print(f"Device {device}, get input time: {time6 - time5}")
    return inputs

class SToFM_Collator():
    def __init__(self, norm_type=0, cls_type=1, hyper_type=2, pad_type=3, pad_indices=-1,
                 mask=False, mask_rate=0.12, remaining_rate=0.03, mask_token=-100.0,
                 mask_pair=False, mask_pair_rate=0.12, remaining_pair_rate=0.03, mask_pair_token=-100.0):
        self.norm_type = norm_type
        self.cls_type = cls_type
        self.hyper_type = hyper_type
        self.pad_type = pad_type
        self.pad_indices = pad_indices

        self.mask = mask
        assert mask_rate + remaining_rate <= 1
        self.mask_rate = mask_rate
        self.remaining_rate = remaining_rate
        self.mask_token = mask_token

        self.mask_pair = mask_pair
        assert mask_pair_rate + remaining_pair_rate <= 1
        self.mask_pair_rate = mask_pair_rate
        self.remaining_pair_rate = remaining_pair_rate
        self.mask_pair_token = mask_pair_token


    def mask_nodes(self, token_embeddings, token_types):
        # token_embeddings: [num_nodes, hidden_size], token_types: [num_nodes]
        num_nodes_norm = (token_types == self.norm_type).sum().item()
        num_masked_nodes = int(self.mask_rate * num_nodes_norm)
        num_remaining_nodes = int(self.remaining_rate * num_nodes_norm)
        random_indices = torch.randperm(num_nodes_norm) + (token_types.shape[0] - num_nodes_norm)
        masked_indices = random_indices[:num_masked_nodes]
        remaining_indices = random_indices[num_masked_nodes:num_masked_nodes + num_remaining_nodes]
        all_indices = torch.cat([masked_indices, remaining_indices])
        all_indices_mask = torch.zeros_like(token_types, dtype=torch.bool)
        all_indices_mask[all_indices] = True
        labels = token_embeddings.clone().detach()
        labels[~all_indices_mask] = self.mask_token
        masked_token_embeddings = token_embeddings.clone().detach()
        masked_token_embeddings[masked_indices] = self.mask_token
        return masked_token_embeddings, labels
    
    def mask_attn_bias(self, attn_bias, node_coops, token_types):
        # attn_bias: [num_nodes, num_nodes], node_coops: [num_nodes, 2], token_types: [num_nodes]
        num_nodes_norm = (token_types == self.norm_type).sum().item()
        num_masked_nodes = int(self.mask_rate * num_nodes_norm)
        num_remaining_nodes = int(self.remaining_rate * num_nodes_norm)
        random_indices = torch.randperm(num_nodes_norm) + (token_types.shape[0] - num_nodes_norm)
        masked_indices = random_indices[:num_masked_nodes]
        remaining_indices = random_indices[num_masked_nodes:num_masked_nodes + num_remaining_nodes]
        all_indices = torch.cat([masked_indices, remaining_indices])

        all_indices_mask = torch.zeros_like(attn_bias, dtype=torch.bool) # [num_nodes, num_nodes] True: masked
        all_indices_mask[all_indices, :] = True
        all_indices_mask[:, all_indices] = True
        for indice in all_indices:
            all_indices_mask[indice, indice] = False

        labels = attn_bias.clone().detach()
        labels[~all_indices_mask] = self.mask_token
        labels[labels == 100.] = self.mask_token
        labels[token_types != self.norm_type] = self.mask_token
        labels[:, token_types != self.norm_type] = self.mask_token

        # add gussian noise to the masked nodes coops
        # set sigma = average distance * 5
        norm_node_coops = node_coops[token_types == self.norm_type]
        x_len = torch.max(norm_node_coops[:, 0]) - torch.min(norm_node_coops[:, 0])
        y_len = torch.max(norm_node_coops[:, 1]) - torch.min(norm_node_coops[:, 1])
        sigma = torch.sqrt(x_len * y_len / num_nodes_norm) * 5
        masked_node_coops = node_coops.clone().detach()
        masked_node_coops[masked_indices] += torch.randn_like(masked_node_coops[masked_indices]) * sigma

        changed_distances = cdist(masked_node_coops[masked_indices], masked_node_coops, metric='euclidean') # [num_masked_nodes, num_nodes]
        # remain top 50
        changed_distances = torch.from_numpy(changed_distances).to(torch.float32)

        new_attn_bias = attn_bias.clone().detach()
        new_attn_bias[masked_indices, :] = changed_distances
        new_attn_bias[:, masked_indices] = changed_distances.T
        new_attn_bias[:, 0] = 0.
        new_attn_bias[0, :] = 0.
        new_attn_bias[new_attn_bias == 100.] += (new_attn_bias.T[new_attn_bias == 100.] - 100) * (new_attn_bias.T[new_attn_bias == 100.] != 100)

        return new_attn_bias, labels 

    def __call__(self, features): # List[Dict[str, Any]]
        # keys: ['spatial', 'token_embeddings', 'attn_bias', 'token_types']
        batch_size = len(features)
        max_nodes = max([f['token_embeddings'].shape[0] for f in features])
        hidden_size = features[0]['token_embeddings'].shape[1]
        batch = {}
        batch['spatial'] = torch.zeros(batch_size, max_nodes, 2, dtype=torch.float32)
        batch['token_embeddings'] = torch.zeros(batch_size, max_nodes, hidden_size, dtype=torch.float32)
        batch['attn_bias'] = torch.zeros(batch_size, max_nodes, max_nodes, dtype=torch.float32)
        batch['token_types'] = torch.ones(batch_size, max_nodes, 
                                 dtype=torch.int32) * self.pad_type
        batch['indices'] = torch.ones(batch_size, max_nodes, 
                                 dtype=torch.int32) * self.pad_indices
        if self.mask:
            batch['labels'] = torch.ones(batch_size, max_nodes, hidden_size, dtype=torch.float32) * self.mask_token
        if self.mask_pair:
            batch['pair_labels'] = torch.ones(batch_size, max_nodes, max_nodes, dtype=torch.float32) * self.mask_pair_token
        for i, f in enumerate(features):
            num_nodes = f['token_embeddings'].shape[0]
            
            if self.mask:
                token_embeddings, labels = self.mask_nodes(f['token_embeddings'], f['token_types'])
                batch['token_embeddings'][i, :num_nodes] = token_embeddings
                batch['labels'][i, :num_nodes] = labels
            else:
                batch['token_embeddings'][i, :num_nodes] = f['token_embeddings']

            if self.mask_pair:
                attn_bias, pair_labels = self.mask_attn_bias(f['attn_bias'], f['spatial'], f['token_types'])
                batch['attn_bias'][i, :num_nodes, :num_nodes] = attn_bias
                batch['pair_labels'][i, :num_nodes, :num_nodes] = pair_labels
            else:
                batch['attn_bias'][i, :num_nodes, :num_nodes] = f['attn_bias']

            batch['spatial'][i, :num_nodes] = f['spatial']
            batch['token_types'][i, :num_nodes] = f['token_types']
            batch['indices'][i, :num_nodes] = f['indices']
        return batch
    
