import pandas as pd
from biodem.utils_model import read_config, dataloader_and_model_pred

def predict_pheno(path_save_result:str, path_model:str, paths_omics: list[str], batch_size: int = 8):
    ## Read config FIRST
    path_config = path_model + ".cfg"
    config = read_config(path_config)
    ### regr_or_clas:int
    regr_or_clas = True
    if config["dim_out"] > 1:
        regr_or_clas = False
    ## RUN prediction
    y = dataloader_and_model_pred(path_model, paths_omics, regr_or_clas, batch_size)
    y = pd.DataFrame(y)
    ## Set colnames: Read label template
    colnamex = pd.read_csv((path_model + "_" + "template_label.csv"), index_col=0)
    y.columns = colnamex.columns.tolist()
    ## Set rownames
    rownamex = pd.read_csv(paths_omics[0], usecols=[0])
    y.index = rownamex.index.tolist()
    ## Write
    y.to_csv(path_save_result)
