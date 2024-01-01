import pandas as pd
import numpy as np
import copy
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
#torch.utils.data as data, torchvision as tv,
from torch.nn import Module
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from lightning import Trainer, LightningModule, LightningDataModule
from lightning.pytorch.callbacks import EarlyStopping

import multiprocessing

n_threads = round(multiprocessing.cpu_count() * 0.9)
print(f'Using {n_threads} threads')

torch.set_float32_matmul_precision('medium')


# from utils_model import dataloader_trte

# def pearson_correlation(tensor1: torch.Tensor, tensor2: torch.Tensor):
#     assert tensor1.shape == tensor2.shape
#     mean1 = torch.mean(tensor1)
#     mean2 = torch.mean(tensor2)
#     std1 = torch.std(tensor1)
#     std2 = torch.std(tensor2)
#     # Calc PCC
#     numerator = ((tensor1 - mean1) * (tensor2 - mean2)).sum()
#     denominator = (std1 * std2).clamp(min=1e-8)# Avoid errors in dividing by 0
#     pearson_correlation = numerator / denominator
#     return pearson_correlation

# def pearson_correlation(tensor1: torch.Tensor, tensor2: torch.Tensor):
#     assert tensor1.shape == tensor2.shape
#     # Calc PCC
#     pearson_correlation = np.corrcoef(tensor1.cpu().detach().numpy().flatten(), tensor2.cpu().detach().numpy().flatten())
#     return pearson_correlation


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

        # print(num_head, dim_model, self.dim_head)

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
    def __init__(self, dims_in_omics: list[int], n_head: int, n_encoder: int, dim_out: int, dim_hidden: int, dropout: float, learning_rate: float):
        super().__init__()
        self.dims_in_omics = dims_in_omics
        self.n_head = n_head
        self.n_encoder = n_encoder
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.net_cat = TransformerAll(sum(dims_in_omics), n_head, n_encoder, dim_hidden, dim_out, dropout)
        self.net_omics1 = TransformerXOmics(dims_in_omics[0], n_head, n_encoder, dim_hidden, dim_out, dropout)
        self.net_omics2 = TransformerXOmics(dims_in_omics[1], n_head, n_encoder, dim_hidden, dim_out, dropout)
        self.net_integrate = TransformerIntegrate(dims_in_omics[0] + sum(dims_in_omics),
                                                  n_head, n_encoder, dim_hidden, dim_out, dropout)

    def forward(self, omics1: torch.Tensor, omics2: torch.Tensor):
        # in lightning, forward defines the prediction/inference actions
        output_cat, hidden_cat = self.net_cat(torch.cat([omics1, omics2], dim=2))
        output_omics1, hidden_omics1 = self.net_omics1(omics1)
        output_omics2, hidden_omics2 = self.net_omics2(omics2)
        hidden_integrate = torch.cat([hidden_cat, hidden_omics1, hidden_omics2], dim=1)
        hidden_integrate = hidden_integrate.reshape(hidden_integrate.shape[0], 1, hidden_integrate.shape[1])
        output_integration = self.net_integrate(hidden_integrate)
        return output_integration, output_cat, output_omics1, output_omics2

    def training_step(self, batch: list[torch.Tensor], batch_idx):
        # training_step defines the train loop. It is independent of forward
        # loss
        loss = nn.L1Loss(reduction="sum")
        # Get x and y
        y, x1, x2 = batch

        # y
        pheno = y.reshape(-1, 1)
        
        # Run
        pheno_hat_integr, pheno_hat_cat, pheno_hat_omics1, pheno_hat_omics2 = self(x1, x2)
        
        # loss
        loss_integration = loss(pheno_hat_integr, pheno)
        total_loss = 1.0 * loss_integration + loss(pheno_hat_cat, pheno) + loss(pheno_hat_omics1, pheno) + loss(pheno_hat_omics2, pheno)
        
        # Pearson correlation coefficient between predicted phenotypes and true phenotypes
        # final_pcc = pearson_correlation(pheno_hat_integr, pheno)

        # self.log("train_loss", total_loss)
        self.log_dict({'train_loss': total_loss, #'train_acc': final_pcc,
                       'train_loss_final': loss_integration})
        return total_loss

    def validation_step(self, batch: list[torch.Tensor], batch_idx):
        # loss
        loss = nn.L1Loss(reduction="sum")
        # Get x and y
        y, x1, x2 = batch

        # y
        pheno = y.reshape(-1, 1)
        
        # Run
        pheno_hat_integr, pheno_hat_cat, pheno_hat_omics1, pheno_hat_omics2 = self(x1, x2)
        
        print(" \nValidation: ", pheno_hat_integr.shape, pheno.shape)
        print(torch.cat([pheno_hat_integr, pheno], dim=1), "\n")
        
        # loss
        loss_integration = loss(pheno_hat_integr, pheno)
        total_loss = 1.0 * loss_integration + loss(pheno_hat_cat, pheno) + loss(pheno_hat_omics1, pheno) + loss(pheno_hat_omics2, pheno)
        
        # Pearson correlation coefficient between predicted phenotypes and true phenotypes
        # final_pcc = pearson_correlation(pheno_hat_integr, pheno)

        self.log_dict({'val_loss': total_loss, #'val_acc': final_pcc,
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
        self.omics1 = pd.read_csv(paths_omics[0], index_col=0).to_numpy(np.float32)
        self.omics2 = pd.read_csv(paths_omics[1], index_col=0).to_numpy(np.float32)

        self.omics1 = self.omics1.reshape(self.omics1.shape[0], 1, self.omics1.shape[1])
        self.omics2 = self.omics2.reshape(self.omics2.shape[0], 1, self.omics2.shape[1])

        self.labels_len = len(self.labels)
        if regr_or_class:
            self.labels = self.labels.flatten()
        else:
            self.labels = LabelEncoder().fit_transform(self.labels)

    def __len__(self):
        return self.labels_len
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        omics1 = self.omics1[idx, :]
        omics2 = self.omics2[idx, :]
        # inputs_cat = torch.cat([torch.Tensor(x) for x in inputs], 1)
        return label, omics1, omics2

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


# -------------------
# Step 3: Train
# -------------------

def dem_train(paths_omics_i: list[str], path_pheno_i: str,
              #   path_model_o: str, 
              regr_clas: bool,
              prop_val: float = 0.2, split_seed: int = 1234,
              batch_size: int = 32, lr: float = 0.0001, dropout: float = 0.1,
              epoch_max: int = 1000, patience: int = 5, n_encoder: int = 4,
              ):
    data_module = DEMDataModule(path_pheno_i, paths_omics_i, regr_clas, prop_val, split_seed, batch_size, n_threads)

    dem = LitDEM([1000, 1000], 5, n_encoder, 1, 1000, dropout, lr)
    early_stopping = EarlyStopping('val_loss', patience=patience, verbose=True)
    trainer = Trainer(precision='16-mixed',
                      accelerator='gpu',
                      devices=1,
                      callbacks=[early_stopping],
                      max_epochs=epoch_max,
                      fast_dev_run=True,
                      )

    trainer.fit(model=dem, datamodule=data_module)


# TRY
paths_omics = ['/home/wuch/prjs/dem/data/preprocessed/01_arabidopsis/FT/mrna_1000.csv',
               '/home/wuch/prjs/dem/data/preprocessed/01_arabidopsis/FT/r01_snp_RF_1000.csv']
path_pheno = '/home/wuch/prjs/dem/data/preprocessed/01_arabidopsis/FT/FT_phenotype_std.csv'
dem_train(paths_omics, path_pheno, True)
