r"""
Default values.
"""
from os import getenv
from numpy import ceil
from multiprocessing import cpu_count


n_threads = int(getenv("NUM_THREADS", ceil(cpu_count() * 0.8)))

time_format = "%Y%m%d%H%M%S"
time_delay = 11.7
ckpt_fname_format = "best-model-{epoch:04d}-{val_loss:.4f}"
optuna_db = "sqlite:///optuna.db"
n_jobs = 1
n_trials = 10
n_workers = 1
n_workers_litdata = 1
accelerator = "auto"
devices = "auto"
float32_matmul_precision = "high"
compression_alg = "zstd"
chunk_bytes = "256MB"

n_jobs_rf = n_threads
variance_threshold = float(getenv("VARIANCE_THRESHOLD", "0.01"))
n_estimators = int(getenv("NUM_ESTIMATORS", "5000"))
n_feat2save = int(getenv("NUM_FEAT2SAVE", "1000"))
random_states = [i + 45 for i in range(10)]

seed_1 = int(getenv("SEED_1", "42"))
seed_2 = int(getenv("SEED_2", "43"))

lr = 1e-4
batch_size = 32
max_epochs = 1000
min_epochs = 20
patience = 20
dropout = 0.4

hidden_dim = 512
n_encoders = 2
n_heads = 2

snp_onehot_bits = 10
