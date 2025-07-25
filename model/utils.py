from transformers.configuration_utils import PretrainedConfig
import torch
from geneformer import DataCollatorForCellClassification
from geneformer.collator_for_classification import PrecollatorForGeneAndCellClassification
from geneformer.pretrainer import token_dictionary
token_dictionary['<cls>'] = len(token_dictionary)

class CellEncoderTokenizer(PrecollatorForGeneAndCellClassification):
    cls_token = "<cls>"
    cls_token_id = token_dictionary.get("<cls>")
    all_special_ids = [
        token_dictionary.get('<cls>'),
        token_dictionary.get("<mask>"),
        token_dictionary.get("<pad>"),
    ]
    token_dictionary = token_dictionary

    def _convert_token_to_id_with_added_voc(self, token):
        if token is None:
            return None

        return self.token_dictionary.get(token)

    def __len__(self):
        return len(self.token_dictionary)
    
class CellEncoderCollator(DataCollatorForCellClassification):
    def __init__(self, add_cls=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_cls = add_cls
        self.tokenizer = CellEncoderTokenizer()

    def _prepare_batch(self, features):
        if self.add_cls:
            for i in range(len(features)):
                features[i]['input_ids'] = ([int(self.tokenizer.cls_token_id)] + features[i]['input_ids'])[:2048]
        
        batch = super()._prepare_batch(features)
        
        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        # (it should be automatically the case, but let's make sure of it.)
        first = features[0]
        if "label" in first and first["label"] is not None:
            if isinstance(first["label"], torch.Tensor):
                label = first["label"].item()
            elif isinstance(first["label"], list):
                label = first["label"][0]
            else:
                label = first["label"]
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
            
        return batch

class SToFMConfig(PretrainedConfig):

    def __init__(
        self,
        num_hidden_layers: int = 12,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 768,
        num_attention_heads: int = 32,
        gaussian_hidden_dim: int = 128,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        layerdrop: float = 0.0,
        encoder_normalize_before: bool = False,
        pre_layernorm: bool = False,
        apply_init: bool = False,
        activation_fn: str = "gelu",
        embed_scale: float = None,
        freeze_embeddings: bool = False,
        num_trans_layers_to_freeze: int = 0,
        traceable: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        kdim: int = None,
        vdim: int = None,
        bias: bool = True,
        self_attention: bool = True,
        norm_type_id=0,
        cls_type_id=1,
        hyper_type_id=2,
        pad_type_id=3,
        **kwargs,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.embedding_dim = embedding_dim
        self.hidden_size = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.num_attention_heads = num_attention_heads
        self.gaussian_hidden_dim = gaussian_hidden_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.layerdrop = layerdrop
        self.encoder_normalize_before = encoder_normalize_before
        self.pre_layernorm = pre_layernorm
        self.apply_init = apply_init
        self.activation_fn = activation_fn
        self.embed_scale = embed_scale
        self.freeze_embeddings = freeze_embeddings
        self.num_trans_layers_to_freeze = num_trans_layers_to_freeze
        self.traceable = traceable
        self.q_noise = q_noise
        self.qn_block_size = qn_block_size

        self.kdim = kdim
        self.vdim = vdim
        self.self_attention = self_attention
        self.bias = bias

        self.norm_type_id = norm_type_id
        self.cls_type_id = cls_type_id
        self.hyper_type_id = hyper_type_id
        self.pad_type_id = pad_type_id

        super().__init__(
            **kwargs,
        )

