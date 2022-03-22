#!/usr/bin/env python
# coding: utf-8


import torch
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

# get_ipython().run_line_magic('matplotlib', 'inline')
# from matplotlib import pyplot as plt
# import matplotlib as mpl

from args_parser import get_parser
from model.mars import MARS
from model.experiment_dataset import ExperimentDataset
import warnings
warnings.filterwarnings('ignore')


# # Setting parameters

# Loading default parameters
params, unknown = get_parser().parse_known_args()

# Checking if CUDA device is available
if torch.cuda.is_available() and not params.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
device = 'cuda:0' if torch.cuda.is_available() and params.cuda else 'cpu'
params.device = device

# # Loading dataset
# Loading Cellbench dataset. Datasets are already concatenated and only the subset of genes that appears in both datasets is retained
adata = sc.read_h5ad('benchmark_datasets/cellbench.h5ad')
# X: <class 'numpy.ndarray'>
# X: (4373, 10217)
# obs: <class 'pandas.core.frame.DataFrame'>
# obs: (4373, 2)

sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
sc.pp.scale(adata, max_value=10, zero_center=True)

sc.pp.neighbors(adata, n_neighbors=30, use_rep='X')
sc.pp.pca(adata, n_comps=50)
# sc.tl.tsne(adata)
# sc.pl.tsne(adata, color=['experiment','ground_truth'],size=50)

datasets = list(set(adata.obs['experiment']))
# # Train and evaluate MARS

# ### Use 10x as annotated, and CelSeq2 as unannotated 

# Prepare annotated, unannotated and pretrain datasets
exp_10x = adata[adata.obs['experiment'] == '10x_5cl',:]
exp_celseq2 = adata[adata.obs['experiment'] == 'CelSeq2_5cl',:]

y_10x = np.array(exp_10x.obs['ground_truth'], dtype=np.int64)
annotated = ExperimentDataset(exp_10x.X.toarray(), exp_10x.obs_names, exp_10x.var_names, '10x', y_10x)
# <class 'numpy.ndarray'>
# <class 'pandas.core.indexes.base.Index'>
# <class 'pandas.core.indexes.base.Index'>, indices, 0, 1, 2, 3, 4, ...

y_celseq2 = np.array(exp_celseq2.obs['ground_truth'], dtype=np.int64) # ground truth annotations will be only used for evaluation
unannnotated = ExperimentDataset(exp_celseq2.X.toarray(), exp_celseq2.obs_names, exp_celseq2.var_names, 'celseq2', y_celseq2)

pretrain_data = ExperimentDataset(exp_celseq2.X.toarray(), exp_celseq2.obs_names, exp_celseq2.var_names, 'celseq2')

n_clusters = len(np.unique(unannnotated.y))


# Initialize MARS
mars = MARS(n_clusters, params, [annotated], unannnotated, pretrain_data, hid_dim_1=1000, hid_dim_2=100)


# Run MARS in evaluation mode. Ground truth annotations will be used to evaluate MARS performance and scores will be returned
# return only unannotated dataset with save_all_embeddings=False
adata, landmarks, scores = mars.train(evaluation_mode=True, save_all_embeddings=False) # evaluation mode

# Check MARS performance
print(scores)

'''
# Visualize in MARS embedding space
#create anndata object using MARS embeddings as X
adata_mars = AnnData(adata.obsm['MARS_embedding'])
adata_mars.obs['MARS_labels'] = pd.Categorical(adata.obs['MARS_labels'])
adata_mars.obs['ground_truth'] = pd.Categorical(adata.obs['truth_labels'])

np.shape(adata_mars.X) # 100-dimensional MARS embeddings space

# visualize only unannotated dataset
# sc.pp.neighbors(adata_mars, n_neighbors=30, use_rep='X')
# sc.tl.umap(adata_mars)
# sc.pl.umap(adata_mars, color=['ground_truth','MARS_labels'],size=50)
'''
