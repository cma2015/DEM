r"""
Hyperparameter candidates.
"""

batch_size = [16, 32, 64, 128]
lr = [1e-3, 1e-4, 1e-5]
dropout_high = 0.8
dropout_step = 0.2

n_heads = [1, 2, 4]
n_encoders = [1, 2, 4]
hidden_dim = [256, 512, 1024]

s2g_dense_layer_dims = [[1024, 512, 128]]

linear_dims_single_omics = [[512, 128]]
linear_dims_conc_omics = [[1536, 512]]
linear_dims_integrated = [[1024, 256, 128]]
