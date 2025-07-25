from geneformer import TranscriptomeTokenizer
from geneformer.tokenizer import tokenize_cell
import scanpy as sc, pickle as pkl, numpy as np

class SToFMTranscriptomeTokenizer(TranscriptomeTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tokenize_anndata(self, data):
        if self.custom_attr_name_dict is not None:
            file_cell_metadata = {
                attr_key: [] for attr_key in self.custom_attr_name_dict.keys()
            }

        # with lp.connect(str(loom_file_path)) as data:
            # define coordinates of detected protein-coding or miRNA genes and vector of their normalization factors
        coding_miRNA_loc = np.where(
            [self.genelist_dict.get(i, False) for i in data.var["ensembl_id"]]
        )[0]
        norm_factor_vector = np.array(
            [
                self.gene_median_dict[i]
                for i in data.var["ensembl_id"][coding_miRNA_loc]
            ]
        )
        coding_miRNA_ids = data.var["ensembl_id"][coding_miRNA_loc]
        coding_miRNA_tokens = np.array(
            [self.gene_token_dict[i] for i in coding_miRNA_ids]
        )

        # define coordinates of cells passing filters for inclusion (e.g. QC)
        try:
            data.obs["filter_pass"]
        except AttributeError:
            var_exists = False
        else:
            var_exists = True

        if var_exists is True:
            filter_pass_loc = np.where(
                [True if i == 1 else False for i in data.obs["filter_pass"]]
            )[0]
        elif var_exists is False:
            print(
                f"data has no column attribute 'filter_pass'; tokenizing all cells."
            )
            filter_pass_loc = np.array([i for i in range(data.shape[0])])

        # scan through .loom files and tokenize cells
        tokenized_cells = []

        # for (_ix, _selection, view) in data.scan(items=filter_pass_loc, axis=1):
        #     # select subview with protein-coding and miRNA genes
        #     subview = view.view[coding_miRNA_loc, :]
        subview = data[filter_pass_loc, coding_miRNA_loc]

        # normalize by total counts per cell and multiply by 10,000 to allocate bits to precision
        # and normalize by gene normalization factors
        subview_norm_array = (
            subview.X.toarray().T
            / subview.obs.n_counts.to_numpy()
            * 10_000
            / norm_factor_vector[:, None]
        )
        # tokenize subview gene vectors
        tokenized_cells += [
            tokenize_cell(subview_norm_array[:, i], coding_miRNA_tokens)
            for i in range(subview_norm_array.shape[1])
        ]

        # add custom attributes for subview to dict
        if self.custom_attr_name_dict is not None:
            for k in file_cell_metadata.keys():
                file_cell_metadata[k] += subview.obs[k].tolist()
        else:
            file_cell_metadata = None

        return tokenized_cells, file_cell_metadata


mouseid2humanid = pkl.load(open("mouseid2humanid.pkl", "rb"))
tk = SToFMTranscriptomeTokenizer({}, nproc=4)
adata = sc.read_h5ad(f'/path/to/adata.h5ad')

adata.var['ensembl_id'] = [mouseid2humanid[gene_id] if gene_id in mouseid2humanid else gene_id for gene_id in adata.var['gene_ids']]
adata.obs['n_counts'] = adata.X.sum(axis=1)
adata.obs['filter_pass'] = True
tokenized_cells, cell_metadata = tk.tokenize_anndata(adata)
tokenized_dataset = tk.create_dataset(tokenized_cells, cell_metadata)
tokenized_dataset.save_to_disk(f"/path/to/hf.dataset")
adata.write(f"/path/to/data.h5ad")