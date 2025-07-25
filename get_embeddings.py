import scanpy as sc, torch, numpy as np
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from model.se2transformer import SToFMModel, SToFMConfig, SToFMForMaskedLM
from model.extraction import load_data, encode_cell, gather_graphs, SToFM_Collator
from transformers import BertModel
from tqdm import tqdm
import os
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cell_encoder_path", type=str, default='/path/to/')
parser.add_argument("--config_path", type=str, default='/path/to/')
parser.add_argument("--model_path", type=str, default='/path/to/')
parser.add_argument("--data_path", type=str, default='/path/to/')
parser.add_argument("--output_filename", type=str, default='stofm_emb.npy')
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--split_num", type=int, default=1000)
parser.add_argument("--leiden_res", type=float, default=1.0)
parser.add_argument("--leiden_alpha", type=float, default=0.2)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

local_rank = 0
is_master = True
device = torch.device("cuda", local_rank)

cell_encoder_path = args.cell_encoder_path
config_path = args.config_path
model_path = args.model_path
data_path = args.data_path
output_filename = args.output_filename
batch_size = args.batch_size
split_num = args.split_num
leiden_res = args.leiden_res
leiden_alpha = args.leiden_alpha
seed = args.seed

random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

config = SToFMConfig.from_pretrained(config_path)
model = SToFMModel(config).to(device)
state_dict = torch.load(model_path)
# state_dict = {'.'.join(k.split('.')[1:]): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)

class Pooler(nn.Module):
    def __init__(self, config, pretrained_proj, proj_dim):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, proj_dim)
        self.proj.load_state_dict(torch.load(pretrained_proj))
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled_output = hidden_states[:, 0]
        pooled_output = F.normalize(self.proj(pooled_output), dim=-1)
        return pooled_output

cell_encoder = BertModel.from_pretrained(f'{cell_encoder_path}/cell_bert')
cell_encoder.pooler = Pooler(cell_encoder.config, pretrained_proj=f'{cell_encoder_path}/cell_proj.bin', proj_dim=256)
cell_encoder = cell_encoder.to(device)

data_infos = []
if data_path.endswith('.txt'):
    dataset_paths = open(data_path).read().strip().split('\n')
    dataset_paths = [path.strip() for path in dataset_paths]
elif data_path.endswith('.allfiles'):
    dataset_paths = list(os.listdir(data_path[:-9]))
    dataset_paths.sort()
    dataset_paths = [f"{data_path[:-9]}/{path}" for path in dataset_paths]
else:
    dataset_paths = data_path.split(',')
for path in dataset_paths:
    data_infos.append({"data_root": f"{path}",
                    "data_path": f"{path}/{'data.h5ad'}", 
                    "spatial_path": None, 
                    "model_input_path": f"{path}/hf.dataset",
                    "emb_path": f"{path}/ce_emb.npy",})
    
step = 0
for data_info in data_infos:
    print(f"Encode cell {data_info['data_path']}")
    if not os.path.exists(data_info['emb_path']) and is_master:
        print(f"Rank {local_rank}, Encode {data_info['data_path']}")
        encode_cell(cell_encoder, data_info['model_input_path'], data_info['emb_path'], save=True, batch_size=32)

    print(f"Load {data_info['data_path']}")
    assert os.path.exists(data_info['emb_path'])
    graphs = load_data(**data_info, new_emb=False, device=local_rank, filter=False, 
                       split_num=split_num, leiden_res=leiden_res, alpha=leiden_alpha)
    if data_info['data_path'].endswith(".h5ad"):
        adata = sc.read_h5ad(data_info['data_path'])
    else:
        adata = sc.read_10x_mtx(data_info['data_path'])
    data_num = len(adata)
    del adata

    print(f"Cell {data_num}, Sub-slice {len(graphs)}")
            
    dataloader = DataLoader(graphs, collate_fn=SToFM_Collator(mask=False, mask_pair=False), 
                            batch_size=batch_size, shuffle=False)
    
    embeddings = torch.zeros(data_num, 256)
    for i, graph in tqdm(enumerate(dataloader), desc=f"Get embedding"):
        step += 1
        indices = graph['indices']
        graph = {k: v.to(device) for k, v in graph.items()}
        model.eval()
        torch.cuda.empty_cache()
        output = model(**graph)
        node_rep = output['last_hidden_state']
        embeddings[indices[indices != -1]] = node_rep[indices != -1].clone().detach().cpu()

        del graph, output, node_rep
    del dataloader, graphs
    torch.cuda.empty_cache()

    np.save(f"{data_info['data_root']}/{output_filename}", embeddings.cpu().numpy())