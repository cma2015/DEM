import pandas as pd
import numpy as np
# import math
import copy
import random
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataset as Dateset
import torch.utils.data.dataloader as Dataloader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float16)

config_cfg = {
    "epoch_max": 550,
    "patience": 30,
    "learning_rate": 0.00001,    
    "n_encoder": 4,
    "dropout": 0.1,
    "dim_out": 1,
}

def save_config(dict_config:dict, path_save_model:str):
    path_w = path_save_model
    try:
        file_o = open(path_w, 'wt')
        file_o.write(json.dumps(dict_config))
        file_o.close()
    except:
        print("Unable to write config to file")

def read_config(path_config:str):
    if not os.path.exists(path_config):
        print("Config file not found")
        try:
            save_config(config_cfg, path_config)
        except:
            print("Unable to create a default config file")
    with open(path_config) as f:
        datacf = f.read()
    global config
    config = json.loads(datacf)
    return config
#config = read_config("config.txt")
integrate_hidden = 1000
all_hidden = 500
num_head = 1


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class Multi_Head_Attention(nn.Module):
	# '''
	# params: dim_model-->hidden dim      num_head
	# '''
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
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

class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product'''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        attention = torch.matmul(Q, K.permute(0, 2, 1))  # Q*K^T
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context

class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
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

class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out


class Transformer_xomics(nn.Module):
    def __init__(self, config, xomic):
        super(Transformer_xomics, self).__init__()
        self.encoder = Encoder(xomic.shape[2], num_head, integrate_hidden, config["dropout"])
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config["n_encoder"])])

        self.fc1 = nn.Linear(xomic.shape[2], 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, config["dim_out"])

    def forward(self, x):
        out = x
        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), -1)
        h_out = F.relu(self.fc1(out))
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        if config["dim_out"] == 1:
            out = self.fc4(out)
        else:
            out = torch.sigmoid(self.fc4(out))
        return out, h_out


class Transformer_all(nn.Module):
    def __init__(self, config, all):
        super(Transformer_all, self).__init__()
        
        self.encoder = Encoder(all.shape[2], num_head, all_hidden, config["dropout"])
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config["n_encoder"])])

        self.fc1 = nn.Linear(all.shape[2], 2000)
        self.fc2 = nn.Linear(2000, 1000)
        self.fc3 = nn.Linear(1000, 500)
        self.fc4 = nn.Linear(500, 100)
        self.fc5 = nn.Linear(100, config["dim_out"])

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
        if config["dim_out"] == 1:
            out = self.fc5(out)
        else:
            out = torch.sigmoid(self.fc5(out))
        return out, h_out


class Transformer_Integrate(nn.Module):
    def __init__(self, config, Integrate):
        super(Transformer_Integrate, self).__init__()
        
        self.encoder = Encoder(Integrate.shape[2], num_head, integrate_hidden, config["dropout"])
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config["n_encoder"])])
        
        self.fc1 = nn.Linear(Integrate.shape[2], 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 100)  
        self.fc4 = nn.Linear(100, 50)
        self.fc5 = nn.Linear(50, config["dim_out"])

    def forward(self, x):
        out = x
        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), -1)  
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        if config["dim_out"] == 1:
            out = self.fc5(out)
        else:
            out = torch.sigmoid(self.fc5(out))
        return out


def MAE(X, Y):
    mae = 0
    for i in range(0, len(X)):
        x = X[i]
        y = Y[i]
        ABS = abs(x-y)
        mae += ABS
    Mae = mae / len(X)
    return Mae

def MSE(X, Y):
    mse = 0
    for i in range(0, len(X)):
        x = X[i]
        y = Y[i]
        square = (x-y) ** 2
        mse += square
    Mse = mse / len(X)
    return Mse


class model_n_omics(nn.Module):
    def __init__(self, config, all_conc, omics_list):
        super(model_n_omics, self).__init__()
        # oms = list(dict_omics.keys())
        # self.oms = oms
        self.net_all = Transformer_all(config, all_conc).to(device)
        self.omics = [Transformer_xomics(config, omics_list[nkey]).to(device) for nkey in range(len(omics_list))]
    def forward(self, all_conc, omics_list):
        all_output, hidden_all = self.net_all(all_conc)
        outputs_and_hiddens = [self.omics[xom](omics_list[xom]) for xom in range(len(omics_list))]
        outputs_and_hiddens = [xtuple for xs in outputs_and_hiddens for xtuple in xs]
        return all_output, hidden_all, *outputs_and_hiddens
    
    def integrate(self, config, hidden_integrate):
        self.net_integrate = Transformer_Integrate(config, hidden_integrate).to(device)
        output_integrate = self.net_integrate(hidden_integrate)
        return output_integrate


class dataset_n_omics(Dateset.Dataset):
    def __init__(self, omics_mx:list, labels):
        self.omics_np = [xx.to_numpy() for xx in omics_mx]
        if type(labels) == str:
            self.labels = ""
        else:
            self.labels = labels
    def __len__(self):
        return len(self.omics_np[0])
    def __getitem__(self, index):
        omics_tensor = [xom[index] for xom in self.omics_np]
        omics_tensor = [torch.Tensor(xom) for xom in omics_tensor]
        if type(self.labels) == str:
            return tuple(omics_tensor)
        else:
            labels = torch.tensor(self.labels[index])
            omics_tensor.append(labels)
            return tuple(omics_tensor)


# def dataloader_test(path_omics: list[str], path_pheno: str, regr_clas: bool, batch_size: int, rand_state: int):
#     # Load omics data
#     omics_and_pheno = []
#     for xi in path_omics:
#         tmp_omic = pd.read_csv(xi, index_col=0)
#         omics_and_pheno.append(tmp_omic)

#     # Load phenotypes
#     pheno = pd.read_csv(path_pheno, index_col=0)
    
#     if regr_clas:
#         pheno_o = pheno
#     else:
#         pheno = pheno.iloc[:,0].to_list()
#         # Encode phenotypes into classes
#         encoder_pheno = LabelEncoder().fit(pheno)
#         pheno_encoded = encoder_pheno.transform(pheno)
#         pheno_o = pheno_encoded.reshape(-1, 1)
    
#     try:
#         pheno_o = pheno_o.to_numpy()
#     except:
#         pass
#     omics_and_pheno.append(pheno_o)

#     n_class = np.unique(pheno_o).shape[0]

#     # Split data into tr and te
#     trte_omics_and_pheno = train_test_split(*omics_and_pheno, test_size=prop_val, random_state=rand_state)

#     dataset_tr = dataset_n_omics(list(trte_omics_and_pheno[:-2][::2]), trte_omics_and_pheno[-2])
#     dataloader_tr = Dataloader.DataLoader(dataset_tr, batch_size=batch_size, shuffle=True)

#     dataset_te = dataset_n_omics(list(trte_omics_and_pheno[:-2][1::2]), trte_omics_and_pheno[-1])
#     dataloader_te = Dataloader.DataLoader(dataset_te, batch_size=batch_size, shuffle=True)
    
#     return dataloader_tr, dataloader_te, n_class



def test_n_omics(dataloader_te, model_te, config: dict, regr_or_clas: bool):
    y_hat_te_list = []

    for a, item_te in enumerate(dataloader_te):
        to_device_omics = [xom.to(device) for xom in item_te]
        to_device_omics = [xom.reshape(xom.shape[0], 1, xom.shape[1]) for xom in to_device_omics]
        all_omics_te = torch.cat(tuple(to_device_omics), 2)
        
        Model = model_n_omics(config, all_omics_te, to_device_omics).to(device)
        Model.load_state_dict(model_te, strict=False)

        outputs_and_hiddens = Model(all_omics_te, to_device_omics)
        hiddens = outputs_and_hiddens[1::2]

        hidden_integrate_te = torch.cat(hiddens, 1)
        hidden_integrate_te = hidden_integrate_te.reshape(hidden_integrate_te.shape[0], 1, hidden_integrate_te.shape[1])
        output_integrate_te = Model.integrate(config, hidden_integrate_te)

        output_integrate_pred = output_integrate_te.cpu()
        output_integrate_pred = output_integrate_pred.detach().numpy()
        if not regr_or_clas:
            y_hat_te_list.extend(np.argmax(output_integrate_pred, axis=1))
        else:
            y_hat_te_list = y_hat_te_list + list(output_integrate_pred)

    return y_hat_te_list


def test_n_omics_sf(model, dataloader, config: dict, regr_or_clas: bool) -> float:
    total_te_loss = 0.0

    for a, item_te in enumerate(dataloader):
        to_device_omics_and_label = [xom.to(device) for xom in item_te]
        labels_te = to_device_omics_and_label[-1].reshape(-1, 1)
        to_device_omics = [xom.reshape(xom.shape[0], 1, xom.shape[1]) for xom in to_device_omics_and_label[:-1]]
        all_omics_te = torch.cat(tuple(to_device_omics), 2)

        Model = model_n_omics(config, all_omics_te, to_device_omics).to(device)
        Model.load_state_dict(model, strict=False)

        outputs_and_hiddens = Model(all_omics_te, to_device_omics)
        hiddens = outputs_and_hiddens[1::2]

        hidden_integrate_te = torch.cat(hiddens, 1)
        hidden_integrate_te = hidden_integrate_te.reshape(hidden_integrate_te.shape[0], 1, hidden_integrate_te.shape[1])
        output_integrate_te = Model.integrate(config, hidden_integrate_te)

        if regr_or_clas:
            loss = nn.L1Loss(reduction='sum').to(device)
        else:
            loss = nn.CrossEntropyLoss(reduction='sum').to(device)
        loss_integrate = loss(output_integrate_te, labels_te)
        total_te_loss += loss_integrate.item()

    return total_te_loss / len(dataloader)


def intersect_omic(path_in:str, path_feats:str):#os.path.join('example_data', 'meth_3000.csv')
    fin = pd.read_csv(path_in, index_col=0)
    fgene:list[str] = list(fin.columns)
    ## Read an omic's features file along with its model
    eg_genes = pd.read_csv(path_feats, header=None)
    eg_genes = eg_genes.transpose().values.tolist()
    eg_genes = eg_genes[0]
    ##
    gene_inters:list[str] = list(np.intersect1d(fgene, eg_genes))
    if len(gene_inters) < 5:
        print("\nError: Too few features!")
        return 1
    if len(gene_inters) == len(eg_genes):
        return fin
    else:
        df_out = pd.DataFrame(0, index=fin.index, columns=eg_genes)
        for gx in gene_inters:
            df_out[gx] = fin[gx]
        return df_out


def dataloader_and_model_pred(path_model: str, paths_omics: list[str], regr_or_clas: bool, batch_size: int):
    model = torch.load(path_model)
    omics_list = [intersect_omic(paths_omics[xom], path_model + "_" + "omics" + str(xom) + "features" + ".csv") for xom in range(len(paths_omics))]

    dataset_te = dataset_n_omics(omics_list, "")
    dataloader_te = Dataloader.DataLoader(dataset_te, batch_size=batch_size, shuffle=False)

    y = test_n_omics(dataloader_te, model, config, regr_or_clas)
    return y


def dataloader_trte(path_omics: list[str], path_pheno: str,
                    prop_val: float, regr_clas: bool,
                    batch_size: int, rand_state: int,
                    shuffle_seed: int | None = None,
                    shuffled_omics_index: int | None = None, shuffled_omics_feat: int | None = None):
    # Load omics data
    omics_and_pheno = []
    for xi in path_omics:
        tmp_omic = pd.read_csv(xi, index_col=0)

        if shuffle_seed is not None and shuffled_omics_index is not None and shuffled_omics_feat is not None:
            if xi == shuffled_omics_index:
                setup_seed(shuffle_seed)
                random.shuffle(tmp_omic.iloc[:, shuffled_omics_feat])

        omics_and_pheno.append(tmp_omic)

    # Load phenotypes
    pheno = pd.read_csv(path_pheno, index_col=0)
    
    if regr_clas:
        pheno_o = pheno
    else:
        pheno = pheno.iloc[:,0].to_list()
        # Encode phenotypes into classes
        encoder_pheno = LabelEncoder().fit(pheno)
        pheno_encoded = encoder_pheno.transform(pheno)
        pheno_o = pheno_encoded.reshape(-1, 1)
    
    try:
        pheno_o = pheno_o.to_numpy()
    except:
        pass
    omics_and_pheno.append(pheno_o)

    n_class = np.unique(pheno_o).shape[0]

    if prop_val > 0.0:
        # Split data into tr and te
        trte_omics_and_pheno = train_test_split(*omics_and_pheno, test_size=prop_val, random_state=rand_state)

        dataset_tr = dataset_n_omics(list(trte_omics_and_pheno[:-2][::2]), trte_omics_and_pheno[-2])
        dataloader_tr = Dataloader.DataLoader(dataset_tr, batch_size=batch_size, shuffle=True)

        dataset_te = dataset_n_omics(list(trte_omics_and_pheno[:-2][1::2]), trte_omics_and_pheno[-1])
        dataloader_te = Dataloader.DataLoader(dataset_te, batch_size=batch_size, shuffle=True)
    
        return dataloader_tr, dataloader_te, n_class
    else:
        dataset_tr = dataset_n_omics(list(omics_and_pheno[:-1]), omics_and_pheno[-1])
        dataloader_tr = Dataloader.DataLoader(dataset_tr, batch_size=batch_size, shuffle=True)
        return dataloader_tr, None, n_class


def train_val_n_omics_regr(dataloader_tr, dataloader_te, path_model:str, epoch_max:int=540, patience:int=20, learning_rate:float=0.00001, dropout: float = 0.1, n_encoder: int = 4):
    try:
        config = read_config(path_model + ".cfg")
    except:
        config = config_cfg
    config["epoch_max"] = epoch_max
    config["patience"] = patience
    config["learning_rate"] = learning_rate
    config["dropout"] = dropout
    config["n_encoder"] = n_encoder
    save_config(config, path_model + ".cfg")
    read_config(path_model + ".cfg")
    
    loss = nn.L1Loss(reduction="sum").to(device)
    total_step_all = 0
    for r, item in enumerate(dataloader_tr):
        to_device_omics = [xom.to(device) for xom in item][:-1]
        to_device_omics = [xom.reshape(xom.shape[0], 1, xom.shape[1]) for xom in to_device_omics]
        all_omics_tr = torch.cat(tuple(to_device_omics), 2)
    
    ## Early stopping
    es_loss:list[float] = []
    ep_best = 1
    
    for i in range(epoch_max):
        print("\n------------ epoch {} begins ------------".format(i + 1))
        labels_tr_list = []
        y_hat_tr_list = []
        total_tr_loss = 0.0
        Model = model_n_omics(config, all_omics_tr, to_device_omics).to(device)
        optimizer_model = torch.optim.Adam(Model.parameters(), lr=learning_rate, eps=1e-3, foreach=False)
        scheduler_model = CosineAnnealingWarmRestarts(optimizer_model, T_0=5, T_mult=2)
        Model.train()

        for a, item in enumerate(dataloader_tr):
            to_device_omics_and_label = [xom.to(device) for xom in item]
            labels_tr = to_device_omics_and_label[-1].reshape(-1, 1)
            to_device_omics = [xom.reshape(xom.shape[0], 1, xom.shape[1]) for xom in to_device_omics_and_label[:-1]]
            all_omics_tr = torch.cat(tuple(to_device_omics), 2)

            outputs_and_hiddens = Model(all_omics_tr, to_device_omics)
            hiddens = outputs_and_hiddens[1::2]

            hidden_integrate_tr = torch.cat(hiddens, 1)
            hidden_integrate_tr = hidden_integrate_tr.reshape(hidden_integrate_tr.shape[0], 1, hidden_integrate_tr.shape[1])
            output_integrate_tr = Model.integrate(config, hidden_integrate_tr)


            outputs = outputs_and_hiddens[::2]
            l_conc_and_omics = [loss(xom, labels_tr) for xom in outputs]
            l_integrate = loss(output_integrate_tr, labels_tr)

            total_loss_tr = sum(l_conc_and_omics) + 1 * l_integrate
            total_tr_loss += total_loss_tr.item()

            output_integrate_pred = output_integrate_tr.cpu()
            output_integrate_pred = output_integrate_pred.detach().numpy()
            y_hat_tr_list = y_hat_tr_list + list(output_integrate_pred)
            labels_tr_list.extend(labels_tr.cpu().detach().numpy())

            optimizer_model.zero_grad()
            total_loss_tr.backward()
            optimizer_model.step()

        scheduler_model.step()
        r, _ = pearsonr(np.array(y_hat_tr_list).reshape(-1,), np.array(labels_tr_list).reshape(-1,))

        total_step_all += 1
        print("---train---")
        print("  loss of train set:{}".format(total_tr_loss))
        print("  R_tr:{}".format(r))

        Model.eval()
        with torch.no_grad():
            labels_te_list = []
            y_hat_te_list = []

            for a, item_te in enumerate(dataloader_te):
                to_device_omics_and_label_te = [xom.to(device) for xom in item_te]
                labels_te = to_device_omics_and_label_te[-1].reshape(-1, 1)

                to_device_omics_te = [xom.reshape(xom.shape[0], 1, xom.shape[1]) for xom in to_device_omics_and_label_te[:-1]]
                all_omics_te = torch.cat(tuple(to_device_omics_te), 2)

                outputs_and_hiddens_te = Model(all_omics_te, to_device_omics_te)
                hiddens_te = outputs_and_hiddens_te[1::2]

                hidden_integrate_te = torch.cat(hiddens_te, 1)
                hidden_integrate_te = hidden_integrate_te.reshape(hidden_integrate_te.shape[0], 1, hidden_integrate_te.shape[1])
                output_integrate_te = Model.integrate(config, hidden_integrate_te)

                output_integrate_pred = output_integrate_te.cpu()
                output_integrate_pred = output_integrate_pred.detach().numpy()
                y_hat_te_list = y_hat_te_list + list(output_integrate_pred)
                labels_te_list.extend(labels_te.cpu().detach().numpy())

            r_val, _ = pearsonr(np.array(y_hat_te_list).reshape(-1,), np.array(labels_te_list).reshape(-1,))
            mae_val = MAE(y_hat_te_list, labels_te_list)
            mse_val = MSE(y_hat_te_list, labels_te_list)

            print("--- val ---")
            print("  r_val:  {}".format(r_val))
            print("  MAE_val:{}".format(mae_val))
            print("  MSE_val:{}".format(mse_val))

            ## ES
            if len(es_loss) == 0:
                es_loss.append(mse_val)
                min_mse_val = mse_val
                best_r = r_val
                model_best = Model.state_dict()
            else:
                if min(es_loss) < mse_val:
                    es_loss.append(mse_val)
                elif min(es_loss) > mse_val:
                    es_loss = [mse_val]
                    model_best = Model.state_dict()
                    min_mse_val = mse_val
                    best_r = r_val
                    ep_best = i + 1
            if len(es_loss) > patience:
                print("\nThe best epoch: ", ep_best)
                print("Minimum MSE of validation: ", min_mse_val, "\n")
                print("Best r of validation: ", best_r)
                # torch.save(model_best, path_model)
                # print("Model saved!", "[{}]".format(path_model))
                break

    torch.save(model_best, path_model)
    print("Model saved!", "[{}]".format(path_model))


def train_val_n_omics_clas(dataloader_tr, dataloader_te,
                           path_model:str,
                           n_class: int,
                           epoch_max:int=540, patience:int=20, learning_rate:float=0.00001, dropout: float = 0.1, n_encoder: int = 4):
    try:
        config = read_config(path_model + ".cfg")
    except:
        config = config_cfg
    config["epoch_max"] = epoch_max
    config["patience"] = patience
    config["learning_rate"] = learning_rate
    config["dim_out"] = n_class
    config["dropout"] = dropout
    config["n_encoder"] = n_encoder
    save_config(config, path_model + ".cfg")
    read_config(path_model + ".cfg")
    
    loss = nn.CrossEntropyLoss(reduction="sum").to(device)
    total_step_all = 0
    
    ## Early stopping
    es_loss:list[float] = []
    ep_best = 1

    for i in range(epoch_max):
        print("\n------------ epoch {} begins ------------".format(i + 1))
        labels_tr_list = []
        y_hat_tr_list = []
        total_tr_loss = 0
        load_times = 1
        
        for a, item in enumerate(dataloader_tr):
            to_device_omics_and_label = [xom.to(device) for xom in item]
            labels_tr = to_device_omics_and_label[-1].reshape(-1)
            to_device_omics_and_label.pop()
            to_device_omics = [xom.reshape(xom.shape[0], 1, xom.shape[1]) for xom in to_device_omics_and_label]
            all_omics_tr = torch.cat(tuple(to_device_omics), 2)

            if load_times == 1:
                Model = model_n_omics(config, all_omics_tr, to_device_omics).to(device)
                optimizer_model = torch.optim.Adam(Model.parameters(), lr=learning_rate, eps=1e-3)
                scheduler_model = CosineAnnealingWarmRestarts(optimizer_model, T_0=5, T_mult=2)
                

            outputs_and_hiddens = Model(all_omics_tr, to_device_omics)
            hiddens = outputs_and_hiddens[1::2]

            hidden_integrate_tr = torch.cat(hiddens, 1)
            hidden_integrate_tr = hidden_integrate_tr.reshape(hidden_integrate_tr.shape[0], 1, hidden_integrate_tr.shape[1])
            output_integrate_tr = Model.integrate(config, hidden_integrate_tr)

            outputs = outputs_and_hiddens[::2]
            l_conc_and_omics = [loss(xom, labels_tr) for xom in outputs]
            l_integrate = loss(output_integrate_tr, labels_tr)

            
            total_loss_tr = sum(l_conc_and_omics) + 1 * l_integrate
            total_tr_loss += total_loss_tr.item()  

            
            output_integrate_pred = output_integrate_tr.cpu()
            output_integrate_pred = output_integrate_pred.detach().numpy()
            y_hat_tr_list.extend(np.argmax(output_integrate_pred, axis=1))
            labels_tr_list.extend(labels_tr.cpu().detach().numpy())

            optimizer_model.zero_grad()
            total_loss_tr.backward()
            optimizer_model.step()

        scheduler_model.step()
        acc_tr = accuracy_score(labels_tr_list, y_hat_tr_list)

        total_step_all += 1
        print("---train---")
        print("  train acc:{}".format(acc_tr))
                
        Model.eval()
        with torch.no_grad():
            labels_te_list = []
            y_hat_te_list = []

            for a, item_te in enumerate(dataloader_te):
                to_device_omics_and_label_te = [xom.to(device) for xom in item_te]
                labels_te = to_device_omics_and_label_te[-1].reshape(-1)

                to_device_omics_te = [xom.reshape(xom.shape[0], 1, xom.shape[1]) for xom in to_device_omics_and_label_te[:-1]]
                all_omics_te = torch.cat(tuple(to_device_omics_te), 2)

                outputs_and_hiddens_te = Model(all_omics_te, to_device_omics_te)
                hiddens_te = outputs_and_hiddens_te[1::2]

                hidden_integrate_te = torch.cat(hiddens_te, 1)
                hidden_integrate_te = hidden_integrate_te.reshape(hidden_integrate_te.shape[0], 1, hidden_integrate_te.shape[1])
                output_integrate_te = Model.integrate(config, hidden_integrate_te)

                output_integrate_pred = output_integrate_te.cpu()
                output_integrate_pred = output_integrate_pred.detach().numpy()
                y_hat_te_list.extend(np.argmax(output_integrate_pred, axis=1))
                labels_te_list.extend(labels_te.cpu().detach().numpy())

            acc_val = accuracy_score(labels_te_list, y_hat_te_list)

            print("--- val ---")
            print("  val acc:{}".format(acc_val))
            
            loss_acc = 1 - acc_val

            ## Early stopping
            if len(es_loss) == 0:
                es_loss.append(loss_acc)
                max_acc_val = acc_val
                model_best = Model.state_dict()
            else:
                if min(es_loss) < loss_acc:
                    es_loss.append(loss_acc)
                elif min(es_loss) > loss_acc:
                    es_loss = [loss_acc]
                    model_best = Model.state_dict()
                    max_acc_val = acc_val
                    ep_best = i + 1
            if len(es_loss) > patience:
                print("\nThe best epoch: ", ep_best)
                print("Maximum Acc of validation: ", max_acc_val, "\n")
                # torch.save(model_best, path_model)
                # print("Model saved!", "[{}]".format(path_model))
                break
            
    torch.save(model_best, path_model)
    print("Model saved!", "[{}]".format(path_model))
