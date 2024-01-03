from biodem.utils_model import*

def save_supplement_for_model(paths_omics: list[str], path_save_model: str):
    for xom in range(len(paths_omics)):
        tmpfile = pd.read_csv(paths_omics[xom], index_col=0)
        tmpcolname = list(tmpfile.columns)
        tmpcoldf = pd.DataFrame(tmpcolname)
        path_w = path_save_model + "_" + "omics" + str(xom) + "features" + ".csv"
        tmpcoldf.to_csv(path_w, header=False, index=False)

def save_supplement_label_template(path_label:str, path_save_model:str):
    dflabel = pd.read_csv(path_label, index_col=0)
    dflabel[:] = ""
    path_w = path_save_model + "_" + "template_label.csv"
    dflabel.to_csv(path_w)


def model_dem(paths_omics_i: list[str], path_pheno_i: str,
              path_model_o: str,
              regr_clas: bool,
              prop_val: float = 0.2, split_seed: int = 1234,
              batch_size: int = 32, lr: float = 0.0001, dropout: float = 0.1,
              epoch_max: int = 1000, patience: int = 15, n_encoder: int = 4):
    # Get dataloader
    dataloader_tr, dataloader_te, n_class = dataloader_trte(paths_omics_i, path_pheno_i, prop_val, regr_clas, batch_size, split_seed)
    
    # Build and train
    if not regr_clas:
        train_val_n_omics_clas(dataloader_tr, dataloader_te, path_model_o, n_class, epoch_max, patience, lr, dropout, n_encoder)
    else:
        train_val_n_omics_regr(dataloader_tr, dataloader_te, path_model_o, epoch_max, patience, lr, dropout, n_encoder)
    
    save_supplement_for_model(paths_omics_i, path_model_o)
    save_supplement_label_template(path_pheno_i, path_model_o)
