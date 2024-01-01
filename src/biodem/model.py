import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from scipy.stats import pearsonr

from torch.utils.data import Dataset, DataLoader, random_split
# import torch.utils.data.dataset as Dateset
# import torch.utils.data.dataloader as Dataloader

import torch, torch.nn as nn, torch.utils.data as data, torchvision as tv, torch.nn.functional as F
from torch.nn import Module
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from lightning import Trainer, LightningModule, LightningDataModule
from lightning.pytorch.callbacks import EarlyStopping

import multiprocessing
n_threads = round(multiprocessing.cpu_count() * 0.9)
print(f'Using {n_threads} threads')

# torch.set_float32_matmul_precision('medium')

# from utils_model import dataloader_trte


# Model ltn modules

class ScaledDotProductAttention(Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, scale=None):
        attention = torch.matmul(Q, K.permute(0, 2, 1))  # Q*K^T
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


class MultiHeadAttention(Module):
    def __init__(self, dim_model: int, num_head: int, dropout: float = 0.1):
        super().__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = ScaledDotProductAttention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)  # reshape to batch*head*sequence_length*(embedding_dim//head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K.size(-1) ** -0.5
        context = self.attention(Q, K, V, scale)
        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out


class PositionWiseFeedForward(Module):
    def __init__(self, dim_model: int, hidden: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out


class Encoder(Module):
    def __init__(self, input_dim: int, n_head: int, output_dim: int, dropout: float):
        super().__init__()
        self.attention = MultiHeadAttention(input_dim, n_head, dropout)
        self.feed_forward = PositionWiseFeedForward(input_dim, output_dim, dropout)
    
    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out


class TransformerXOmics(Module):
    def __init__(self, dim_in: int, n_head: int, n_encoder: int, dim_hidden: int, dim_out: int, dropout: float):
        super().__init__()
        self.dim_out = dim_out
        self.encoder = Encoder(dim_in, n_head, dim_hidden, dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(n_encoder)])

        self.fc1 = nn.Linear(dim_in, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, dim_out)

    def forward(self, x):
        out = x
        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), -1)
        h_out = F.relu(self.fc1(out))
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        if self.dim_out == 1:
            out = self.fc4(out)
        else:
            out = torch.sigmoid(self.fc4(out))
        return out, h_out


class TransformerAll(Module):
    def __init__(self, dim_in: int, n_head: int, n_encoder: int, dim_hidden: int, dim_out: int, dropout: float):
        super().__init__()
        self.dim_out = dim_out
        self.encoder = Encoder(dim_in, n_head, dim_hidden, dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(n_encoder)])

        self.fc1 = nn.Linear(dim_in, 2000)
        self.fc2 = nn.Linear(2000, 1000)
        self.fc3 = nn.Linear(1000, 500)
        self.fc4 = nn.Linear(500, 100)
        self.fc5 = nn.Linear(100, dim_out)

    def forward(self, x):
        out = x
        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), -1)
        h_out = F.relu(self.fc1(out))
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        if self.dim_out == 1:
            out = self.fc5(out)
        else:
            out = torch.sigmoid(self.fc5(out))
        return out, h_out


class TransformerIntegrate(Module):
    def __init__(self, dim_in: int, n_head: int, n_encoder: int, dim_hidden: int, dim_out: int, dropout: float):
        super().__init__()
        self.dim_out = dim_out
        self.encoder = Encoder(dim_in, n_head, dim_hidden, dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(n_encoder)])
        
        self.fc1 = nn.Linear(dim_in, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 100)  
        self.fc4 = nn.Linear(100, 50)
        self.fc5 = nn.Linear(50, dim_out)

    def forward(self, x):
        out = x
        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), -1)  
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        if self.dim_out == 1:
            out = self.fc5(out)
        else:
            out = torch.sigmoid(self.fc5(out))
        return out


class LitDEM(LightningModule):
    def __init__(self, dim_in_cat: int, dims_in_omics: list[int], n_head: int, n_encoder: int, dim_out: int, dim_hidden: int, dropout: float, learning_rate: float):
        super().__init__()
        self.dims_in_omics = dims_in_omics
        self.n_head = n_head
        self.n_encoder = n_encoder
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.net_cat = TransformerAll(dim_in_cat, n_head, n_encoder, dim_hidden, dim_out, dropout)
        self.net_omics = [TransformerXOmics(dimx, n_head, n_encoder, dim_hidden, dim_out, dropout) for dimx in dims_in_omics]
        self.net_integrate = TransformerIntegrate(2000 + sum(dims_in_omics),
                                                  n_head, n_encoder, dim_hidden, dim_out, dropout)

    def forward(self, omics_cat, omics_list):
        # in lightning, forward defines the prediction/inference actions
        output_cat, hidden_cat = self.net_cat(omics_cat)
        outputs_and_hiddens = [self.net_omics[dimxind](omics_list[dimxind]) for dimxind in range(len(self.dims_in_omics))]
        outputs_and_hiddens = [xtuple for xs in outputs_and_hiddens for xtuple in xs]
        # return output_cat, hidden_cat, *outputs_and_hiddens
        
        hiddens = (output_cat, hidden_cat, *outputs_and_hiddens)[1::2]
        hidden_integrate_tr = torch.cat(hiddens, 1)
        hidden_integrate_tr = hidden_integrate_tr.reshape(hidden_integrate_tr.shape[0], 1, hidden_integrate_tr.shape[1])
        output_integration = self.net_integrate(hidden_integrate_tr)

        outputs = (output_cat, hidden_cat, *outputs_and_hiddens)[0::2]

        return output_integration, outputs

    def training_step(self, batch: list[torch.Tensor], batch_idx):
        # training_step defines the train loop. It is independent of forward
        # loss
        loss = nn.L1Loss(reduction="sum")
        # Get x and y
        d_y, d_x = batch
        
        # x
        # print('====================================start')
        # print('======================================end')
        # ====================================start
        # <class 'torch.Tensor'>
        # torch.Size([32])
        # <class 'torch.Tensor'>
        # ======================================end

        omics_list = d_x
        omics_cat = torch.cat(omics_list, 2)
        # y
        pheno = d_y.reshape(-1, 1)
        
        # Run
        pheno_predicted_integr, pheno_predicted_sepra = self(omics_cat, omics_list)
        
        # loss
        losses_cat_and_list = [loss(pheno_predicted_sepra[i], pheno) for i in range(len(pheno_predicted_sepra))]
        loss_integration = loss(pheno_predicted_integr, pheno)
        total_loss = sum(losses_cat_and_list) + 1.0 * loss_integration
        
        # Pearson correlation coefficient between predicted phenotypes and true phenotypes
        final_pcc, _ = pearsonr(pheno_predicted_integr.numpy(), pheno.numpy())

        # self.log("train_loss", total_loss)
        self.log_dict({'train_loss': total_loss, 'train_acc': final_pcc,
                       'train_loss_final': loss_integration})
        return total_loss

    def validation_step(self, batch: list[torch.Tensor], batch_idx):
        # loss
        loss = nn.L1Loss(reduction="sum")
        # Get x and y
        d_y, d_x = batch

        omics_list = d_x
        omics_cat = torch.cat(omics_list, 2)
        # y
        pheno = d_y.reshape(-1, 1)
        
        # Run
        pheno_predicted_integr, pheno_predicted_sepra = self(omics_cat, omics_list)
        
        # loss
        losses_cat_and_list = [loss(pheno_predicted_sepra[i], pheno) for i in range(len(pheno_predicted_sepra))]
        loss_integration = loss(pheno_predicted_integr, pheno)
        total_loss = sum(losses_cat_and_list) + 1.0 * loss_integration
        
        # Pearson correlation coefficient between predicted phenotypes and true phenotypes
        final_pcc, _ = pearsonr(pheno_predicted_integr.numpy(), pheno.numpy())

        self.log_dict({'val_loss': total_loss, 'val_acc': final_pcc,
                       'val_loss_final': loss_integration})
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-3)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
        return {'optimizer': optimizer, 
                'lr_scheduler': {
                    'scheduler': scheduler, 
                    'interval': 'step', 
                    'frequency': 1,
                    'monitor': 'val_loss',
                    }
                }


# -------------------
# Step 2: Define data
# -------------------
# dataset = tv.datasets.MNIST(".", download=True, transform=tv.transforms.ToTensor())
# train, val = data.random_split(dataset, [55000, 5000])

class DEMDataset(Dataset):
    def __init__(self, path_label: str, paths_omics: list[str], regr_or_class: bool):
        self.labels = pd.read_csv(path_label, index_col=0).to_numpy(np.float32)
        self.inputs = [pd.read_csv(file_path, index_col=0).to_numpy(np.float32) for file_path in paths_omics]
        self.inputs = [x.reshape(x.shape[0], 1, x.shape[1]) for x in self.inputs]
        self.labels_len = len(self.labels)
        if regr_or_class:
            self.labels = self.labels#.flatten()
        else:
            self.labels = LabelEncoder().fit_transform(self.labels)

    def __len__(self):
        return self.labels_len
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        inputs = [x[idx, :] for x in self.inputs]
        # inputs_cat = torch.cat([torch.Tensor(x) for x in inputs], 1)
        return label, inputs

class DEMDataModule(LightningDataModule):
    def __init__(self, path_label: str, paths_omics: list[str], regr_or_class: bool,
                 prop_val: float, random_state: int, batch_size: int, num_workers: int):
        super().__init__()
        self.path_label = path_label
        self.paths_omics = paths_omics
        self.regr_or_class = regr_or_class
        self.prop_val = prop_val
        self.random_state = random_state
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage: str):
        if stage == 'fit':
            self.dataset_trn, self.dataset_val = random_split(
                DEMDataset(self.path_label, self.paths_omics, self.regr_or_class),
                [1 - self.prop_val, self.prop_val],
                generator=torch.Generator().manual_seed(self.random_state)
            )
        elif stage == 'test':
            self.dataset_tst = DEMDataset(self.path_label, self.paths_omics, self.regr_or_class)
        elif stage == 'predict':
            self.dataset_prd = DEMDataset(self.path_label, self.paths_omics, self.regr_or_class)
    
    def train_dataloader(self):
        return DataLoader(self.dataset_trn, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers)
    def test_dataloader(self):
        return DataLoader(self.dataset_tst, batch_size=self.batch_size, num_workers=self.num_workers)


paths_omics = ['/home/wuch/prjs/dem/data/preprocessed/01_arabidopsis/FT/mrna_1000.csv',
               '/home/wuch/prjs/dem/data/preprocessed/01_arabidopsis/FT/r01_snp_RF_1000.csv']
path_pheno = '/home/wuch/prjs/dem/data/preprocessed/01_arabidopsis/FT/FT_phenotype_std.csv'
# train_loader, val_loader = dataloader_trte(paths_omics, path_pheno, 0.2, True, 32, 1234)
data_module = DEMDataModule(path_pheno, paths_omics, True, 0.2, 1234, 32, n_threads)

# -------------------
# Step 3: Train
# -------------------
dem = LitDEM(2000, [1000, 1000], 5, 5, 1, 500, 0.1, 1e-3)
early_stopping = EarlyStopping('val_loss')
trainer = Trainer(precision='16-mixed',
                  accelerator='gpu',
                  devices=1,
                  callbacks=[early_stopping], check_val_every_n_epoch=1,
                  fast_dev_run=True)

# train_loader = data.DataLoader(train, batch_size=batch_size, num_workers=n_threads, shuffle=True)
# val_loader = data.DataLoader(val, batch_size=batch_size, num_workers=n_threads)

trainer.fit(model=dem, datamodule=data_module)
