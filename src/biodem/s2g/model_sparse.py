r"""
Module for the DEM model with sparse linear layers.
"""
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch_sparse import SparseTensor
from biodem.utils.uni import get_map_location


class SparseLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, fixed_indices: np.ndarray, map_location: Optional[str]=None):
        r"""A linear layer with sparse weights.
        """
        super(SparseLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fixed_indices = torch.tensor(fixed_indices, dtype=torch.long, device=get_map_location(map_location)).t()
        self.sparse_values = nn.Parameter(torch.randn(self.fixed_indices.shape[1], device=get_map_location(map_location)))

    def forward(self, x: torch.Tensor):
        # Create a sparse tensor from the fixed indices and trainable values
        sparse_weight = SparseTensor(row=self.fixed_indices[0], col=self.fixed_indices[1], value=self.sparse_values,
                                     sparse_sizes=(self.output_dim, self.input_dim), trust_data=True)
        # Perform the sparse matrix multiplication
        out = sparse_weight @ x.t()
        return out.t()


def get_param4sparse(blocks_gt: List[List[int]], snp_onehot_bits: int):
    n_gt = len(np.unique(np.concatenate(blocks_gt)))
    input_dim = n_gt * snp_onehot_bits
    output_dim = len(blocks_gt)

    # Index
    idx_axis_0 = np.array([], dtype=int)
    idx_axis_1 = np.array([], dtype=int)
    for i_b in range(output_dim):
        blocks_indices = np.array(blocks_gt[i_b])[:, np.newaxis] * snp_onehot_bits + np.arange(snp_onehot_bits)
        idx_axis_0 = np.concatenate((idx_axis_0, blocks_indices.flatten()))
        idx_axis_1 = np.concatenate((idx_axis_1, np.full(blocks_indices.size, i_b)))
    
    assert len(idx_axis_0) == len(idx_axis_1)

    indices_ = np.vstack((idx_axis_0, idx_axis_1)).T
    
    assert indices_.flatten().min() >= 0
    assert indices_.flatten().max() < input_dim
    
    return indices_, input_dim, output_dim


class SNPReductionNetModel(nn.Module):
    def __init__(
            self,
            output_dim: int,
            blocks_gt: List[List[int]],
            snp_onehot_bits: int,
            dense_layer_dims: List[int],
        ):
        r""" A model for predicting the phenotype from the genome blocks.
        """
        super().__init__()
        n_blocks = len(blocks_gt)
        self.n_blocks = n_blocks
        
        s_index, s_input_dim, s_output_dim = get_param4sparse(blocks_gt, snp_onehot_bits)
        self.sparse_layer = SparseLinear(s_input_dim, s_output_dim, s_index)

        # Define the dense layers for predicting the phenotype
        self.dense_layers = nn.ModuleList()
        
        # Apply LayerNorm to the input features.
        self.dense_layers.append(nn.LayerNorm(n_blocks))
        
        # First dense layer takes the genome blocks features as input.
        self.dense_layers.append(nn.Linear(n_blocks, dense_layer_dims[0]))
        for i_dim in range(len(dense_layer_dims) - 1):
            self.dense_layers.append(nn.Linear(dense_layer_dims[i_dim], dense_layer_dims[i_dim + 1]))
            self.dense_layers.append(nn.Sigmoid())
            # self.dense_layers.append(nn.Dropout(p=0.1))
        self.dense_layers.append(nn.Linear(dense_layer_dims[-1], output_dim))
    
    def forward(self, x):
        # Map SNPs to genome features
        gblocks = self.sparse_layer(x)
        
        for layer in self.dense_layers:
            gblocks = layer(gblocks)
        
        return gblocks
