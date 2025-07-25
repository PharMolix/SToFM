import math
from typing import Iterable, Iterator, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithNoAttention,
    BaseModelOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from .utils import SToFMConfig


class LayerDropModuleList(nn.ModuleList):
    def __init__(self, p: float, modules: Optional[Iterable[nn.Module]] = None):
        super().__init__(modules)
        self.p = p

    def __iter__(self) -> Iterator[nn.Module]:
        dropout_probs = torch.empty(len(self)).uniform_()
        for i, m in enumerate(super().__iter__()):
            if not self.training or (dropout_probs[i] > self.p):
                yield m

class GaussianModule(nn.Module):
    def __init__(self, config: SToFMConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        K = config.gaussian_hidden_dim
        self.K = K
        self.linear = nn.Linear(1, 1)
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)

        self.proj = nn.Sequential(
            nn.Linear(K, K),
            nn.ReLU(),
            nn.Linear(K, config.num_attention_heads),
        )
    def forward(
        self,
        x: torch.Tensor, # [bs, n_node, n_node]
    ) -> torch.Tensor:
        zero_mask = x.eq(0.)
        x = self.linear(x.unsqueeze(-1))
        x = x.expand(-1, -1, -1, self.K) # [bs, n_node, n_node, K]
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        x = torch.exp(-0.5 * (((x - mean) / std) ** 2)) / ((2 * np.pi) ** 0.5 * std)

        x = self.proj(x) # [bs, n_node, n_node, num_heads]
        x = x.permute(0, 3, 1, 2).contiguous() # [bs, num_heads, n_node, n_node]
        x = x.masked_fill(zero_mask.unsqueeze(1).expand_as(x), 0.)
        return x # [bs, num_heads, n_node, n_node]


class MultiheadAttention(nn.Module):
    def __init__(self, config: SToFMConfig):
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.kdim = config.kdim if config.kdim is not None else config.embedding_dim
        self.vdim = config.vdim if config.vdim is not None else config.embedding_dim
        self.qkv_same_dim = self.kdim == config.embedding_dim and self.vdim == config.embedding_dim

        self.num_heads = config.num_attention_heads
        self.attention_dropout_module = torch.nn.Dropout(p=config.attention_dropout, inplace=False)

        self.head_dim = config.embedding_dim // config.num_attention_heads
        if not (self.head_dim * config.num_attention_heads == self.embedding_dim):
            raise AssertionError("The embedding_dim must be divisible by num_heads.")
        self.scaling = self.head_dim**-0.5

        self.self_attention = True  # config.self_attention
        if self.self_attention and not self.qkv_same_dim:
            raise AssertionError("Self-attention requires query, key and value to be of the same size.")

        self.k_proj = nn.Linear(self.kdim, config.embedding_dim, bias=config.bias)
        self.v_proj = nn.Linear(self.vdim, config.embedding_dim, bias=config.bias)
        self.q_proj = nn.Linear(config.embedding_dim, config.embedding_dim, bias=config.bias)
        self.out_proj = nn.Linear(config.embedding_dim, config.embedding_dim, bias=config.bias)

        self.onnx_trace = False

    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: torch.LongTensor, # [n_node, bs, dim]
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        attn_bias: Optional[torch.Tensor], # [bs, n_head, n_node, n_node]
        key_padding_mask: Optional[torch.Tensor] = None, # [bs, n_node]
        need_weights: bool = True,
        # attn_mask: Optional[torch.Tensor] = None,
        return_pair_rep: bool = True,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embedding_dim = query.size() # n_node, bs, dim
        src_len = tgt_len
        if not (embedding_dim == self.embedding_dim):
            raise AssertionError(
                f"The query embedding dimension {embedding_dim} is not equal to the expected embedding_dim"
                f" {self.embedding_dim}."
            )
        if not (list(query.size()) == [tgt_len, bsz, embedding_dim]):
            raise AssertionError("Query size incorrect, compared to model dimensions.")

        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                if (key_bsz != bsz) or (value is None) or not (src_len, bsz == value.shape[:2]):
                    raise AssertionError(
                        "The batch shape does not match the key or value shapes provided to the attention."
                    )

        q = self.q_proj(query) # [n_node, bs, dim]
        k = self.k_proj(query)
        v = self.v_proj(query)

        q *= self.scaling

        # n_node, bs, dim(head_dim * num_head) -> n_node, bs * num_heads, head_dim 
        # -> bs * num_heads, n_node, head_dim
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if (k is None) or not (k.size(1) == src_len):
            raise AssertionError("The shape of the key generated in the attention is incorrect")

        if key_padding_mask is not None and (key_padding_mask.size(0) != bsz or key_padding_mask.size(1) != src_len):
                print(key_padding_mask.size(), bsz, src_len)
                raise AssertionError(
                    "The shape of the generated padding mask for the key does not match expected dimensions."
                )
        attn_weights = torch.bmm(q, k.transpose(1, 2)) # [bs * num_heads, n_node, n_node]
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        if list(attn_weights.size()) != [bsz * self.num_heads, tgt_len, src_len]:
            raise AssertionError("The attention weights generated do not match the expected dimensions.")
        
        if attn_bias is not None:
            # print(attn_bias.shape, bsz, self.num_heads, tgt_len, src_len)
            attn_weights += attn_bias.contiguous().view(bsz * self.num_heads, tgt_len, src_len)

        # if attn_mask is not None:
        #     attn_mask = attn_mask.unsqueeze(0)
        #     attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf") # [bs, 1, 1, n_node]
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v
        
        pair_rep: torch.Tensor = None
        if return_pair_rep:
            pair_rep = torch.clone(attn_weights)
            pair_rep = pair_rep.masked_fill(pair_rep == -np.inf, 0)
            pair_rep = pair_rep.view(bsz, self.num_heads, tgt_len, src_len) # [bs, num_heads, n_node, n_node]

        attn_weights_float = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.attention_dropout_module(attn_weights)

        if v is None:
            raise AssertionError("No value generated")
        attn = torch.bmm(attn_probs, v)
        if list(attn.size()) != [bsz * self.num_heads, tgt_len, self.head_dim]:
            raise AssertionError("The attention generated do not match the expected dimensions.")

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embedding_dim)
        attn: torch.Tensor = self.out_proj(attn)

        attn_weights = None
        if need_weights:
            attn_weights = attn_weights_float.contiguous().view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights, pair_rep

    def apply_sparse_mask(self, attn_weights: torch.Tensor, tgt_len: int, src_len: int, bsz: int) -> torch.Tensor:
        return attn_weights


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config: SToFMConfig) -> None:
        super().__init__()

        # Initialize parameters
        self.embedding_dim = config.embedding_dim
        self.num_attention_heads = config.num_attention_heads
        self.pre_layernorm = config.pre_layernorm

        self.dropout_module = torch.nn.Dropout(p=config.dropout, inplace=False)

        self.activation_dropout_module = torch.nn.Dropout(p=config.activation_dropout, inplace=False)
        self.activation_fn = ACT2FN[config.activation_fn]
        self.self_attn = MultiheadAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)

        self.fc1 = nn.Linear(
            self.embedding_dim,
            config.ffn_embedding_dim,
        )
        self.fc2 = nn.Linear(
            config.ffn_embedding_dim,
            self.embedding_dim,
        )
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

    def forward(
        self,
        input_nodes: torch.Tensor, # [n_node, bs, dim]
        self_attn_bias: Optional[torch.Tensor] = None, # [bs, n_head, n_node, n_node]
        # self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None, # [bs, n_node]
        need_weights: bool = False,
        no_attn_bias: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = input_nodes
        if self.pre_layernorm:
            input_nodes = self.self_attn_layer_norm(input_nodes)

        if no_attn_bias:
            self_attn_bias = torch.zeros_like(self_attn_bias)

        input_nodes, attn, pair_rep = self.self_attn(
            query=input_nodes,
            key=input_nodes,
            value=input_nodes,
            attn_bias=self_attn_bias,
            key_padding_mask=self_attn_padding_mask,
            need_weights=need_weights,
            return_pair_rep=True
        )
        input_nodes = self.dropout_module(input_nodes)
        input_nodes = residual + input_nodes
        if not self.pre_layernorm:
            input_nodes = self.self_attn_layer_norm(input_nodes)

        residual = input_nodes
        if self.pre_layernorm:
            input_nodes = self.final_layer_norm(input_nodes)
        input_nodes = self.activation_fn(self.fc1(input_nodes))
        input_nodes = self.activation_dropout_module(input_nodes)
        input_nodes = self.fc2(input_nodes)
        input_nodes = self.dropout_module(input_nodes)
        input_nodes = residual + input_nodes
        if not self.pre_layernorm:
            input_nodes = self.final_layer_norm(input_nodes)

        return input_nodes, attn, pair_rep


class TransformerEncoder(nn.Module):
    def __init__(self, config: SToFMConfig):
        super().__init__()

        self.dropout_module = torch.nn.Dropout(p=config.dropout, inplace=False)
        self.layerdrop = config.layerdrop
        self.embedding_dim = config.embedding_dim
        self.apply_init = config.apply_init
        self.traceable = config.traceable

        if config.pre_layernorm:
            self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend([TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)])

        for layer in range(config.num_trans_layers_to_freeze):
            m = self.layers[layer]
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

    def forward(
        self,
        # input_nodes: torch.LongTensor,
        attn_bias: torch.Tensor, # [bs, n_head, n_node, n_node]
        token_embeddings: Optional[torch.Tensor], # [bs, n_node, dim]
        token_types: Optional[torch.LongTensor],
        last_state_only: bool = False,
        padding_token_type: int = 3,
        no_attn_bias = False,
        need_attn = False,
    ) -> Tuple[Union[torch.Tensor, List[torch.LongTensor]], torch.Tensor]:
        # compute padding mask. This is needed for multi-head attention
        input_nodes = token_embeddings # [bs, n_node, dim]
        bs = token_embeddings.shape[0]
        padding_mask = token_types.eq(padding_token_type) # [bs, n_node]
        input_nodes = self.dropout_module(input_nodes)
        input_nodes = input_nodes.transpose(0, 1) # [n_node, bs, dim]
        inner_states = []
        attns = []
        if not last_state_only:
            inner_states.append(input_nodes)
        
        for layer in self.layers:
            input_nodes, attn, pair_rep = layer(
                input_nodes,
                self_attn_padding_mask=padding_mask,
                # self_attn_mask=attn_mask,
                self_attn_bias=attn_bias,
                need_weights=need_attn,
                no_attn_bias=no_attn_bias,
            )
            attn_bias = pair_rep
            if not last_state_only:
                inner_states.append(input_nodes)
            if need_attn:
                attns.append(attn)
        if last_state_only:
            inner_states = [input_nodes]
        return inner_states, attns, pair_rep


class SToFMPreTrainedModel(PreTrainedModel):

    config_class = SToFMConfig

    def normal_(self, data: torch.Tensor):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    def init_params(self, module: Union[nn.Linear, nn.Embedding, MultiheadAttention]):
        if isinstance(module, nn.Linear):
            self.normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            self.normal_(module.weight.data)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if isinstance(module, MultiheadAttention):
            self.normal_(module.q_proj.weight.data)
            self.normal_(module.k_proj.weight.data)
            self.normal_(module.v_proj.weight.data)

    def _init_weights(
        self,
        module: Union[
            nn.Linear, nn.Conv2d, nn.Embedding, nn.LayerNorm, MultiheadAttention, 
            TransformerEncoder, GaussianModule
        ],
    ):
        """
        Initialize the weights
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # We might be missing part of the Linear init, dependant on the layer num
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, MultiheadAttention):
            module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.v_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.reset_parameters()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, TransformerEncoder):
            if module.apply_init:
                module.apply(self.init_params)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, GaussianModule):
            module.means.weight.data.uniform_(0, 3)
            module.stds.weight.data.uniform_(0, 3)
            module.linear.weight.data.fill_(1)
            module.linear.bias.data.zero_()
            module.proj[0].weight.data.normal_(mean=0.0, std=0.02)
            module.proj[0].bias.data.zero_()
            module.proj[2].weight.data.normal_(mean=0.0, std=0.02)
            module.proj[2].bias.data.zero_()


class SToFMModel(SToFMPreTrainedModel):
    def __init__(self, config: SToFMConfig):
        super().__init__(config)
        # self.max_nodes = config.max_nodes

        self.embedding_layer = nn.Linear(config.input_dim, config.embedding_dim)
        self.token_type_embeddings = nn.Embedding(4, config.embedding_dim)
        self.gaussian = GaussianModule(config)
        self.encoder = TransformerEncoder(config)
        self.lm_output_learned_bias = None
        self.load_softmax = not getattr(config, "remove_head", False)
        self.post_init()

    def reset_output_layer_parameters(self):
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        token_embeddings: Optional[torch.Tensor], # [bs, n_node, dim]
        attn_bias: torch.Tensor, # [bs, n_node, n_node]
        token_types: Optional[torch.LongTensor], # [bs, n_node]
        need_attn: Optional[bool] = False,
        no_attn_bias: Optional[bool] = False,
        **unused,
    ) -> Union[Tuple[torch.LongTensor], BaseModelOutputWithNoAttention]:

        token_embeddings = self.embedding_layer(token_embeddings)
        if token_types is not None:
            token_type_embeddings = self.token_type_embeddings(token_types)
            token_embeddings += token_type_embeddings
        attn_bias = self.gaussian(attn_bias)

        inner_states, attns, pair_rep = self.encoder(
            token_embeddings=token_embeddings, token_types=token_types, attn_bias=attn_bias,
            no_attn_bias=no_attn_bias, need_attn=need_attn
        ) # inner_states: [(n_node, bs, dim)]

        input_nodes = inner_states[-1].transpose(0, 1) # [bs, n_node, dim]
        pair_rep = pair_rep.permute(0, 2, 3, 1).contiguous() # [bs, n_node, n_node, num_heads]
        result = {'last_hidden_state': input_nodes, 'hidden_states': inner_states, 'pair_rep': pair_rep}
        if need_attn:
            result['attentions'] = attns
        return result

class SToFMForMaskedLM(SToFMPreTrainedModel):
    def __init__(self, config: SToFMConfig):
        super().__init__(config)
        self.model = SToFMModel(config)
        self.lm_head = nn.Sequential( 
                    nn.Linear(config.hidden_size, config.hidden_size), 
                    nn.ReLU(), 
                    nn.Linear(config.hidden_size, config.hidden_size))
        self.pair_head = nn.Sequential(
                    nn.Linear(config.num_attention_heads, config.num_attention_heads),
                    nn.ReLU(),
                    nn.Linear(config.num_attention_heads, 1))
        

        self.init_weights()

    def forward(
        self,
        # input_nodes: torch.LongTensor,
        token_embeddings: Optional[torch.Tensor], # [bs, n_node, dim]
        attn_bias: torch.Tensor, # [bs, n_node, n_node]
        token_types: Optional[torch.LongTensor], # [bs, n_node]
        labels: Optional[torch.Tensor] = None, # [bs, n_node, dim]
        pair_labels: Optional[torch.Tensor] = None, # [bs, n_node, n_node]
        need_attn: Optional[bool] = False,
        no_attn_bias: Optional[bool] = False,
        mask_token=-100.0,
        pair_mask_token=-100.0,
        **unused,
    ) -> Union[Tuple[torch.LongTensor], BaseModelOutput]:

        outputs = self.model(
            token_embeddings=token_embeddings, attn_bias=attn_bias, token_types=token_types, 
            need_attn=need_attn, no_attn_bias=no_attn_bias
        )

        if labels is not None:
            pred = F.normalize(self.lm_head(outputs['last_hidden_state'][labels[:,:,0] != mask_token]), dim=-1)
            loss_fct = nn.CosineEmbeddingLoss()
            loss = loss_fct(pred, labels[labels[:,:,0] != mask_token], torch.ones(len(pred)).to(pred.device))
            outputs['logits'] = pred
            outputs['loss'] = loss

        if pair_labels is not None:
            pair_masked_pos = (pair_labels != pair_mask_token)# [bs, n_node, n_node]


            pair_labels = pair_labels[pair_masked_pos] # [masked_pair_num]
            pair_rep_pred = outputs['pair_rep'][pair_masked_pos] # [masked_pair_num, num_heads]
            pair_pred = self.pair_head(pair_rep_pred).squeeze(-1) # [masked_pair_num]
            
            pair_loss_fct = nn.MSELoss()
            pair_loss = pair_loss_fct(pair_pred, pair_labels)
            outputs['pair_pred'] = pair_pred
            outputs['pair_loss'] = pair_loss

        return outputs