import copy
import numpy as np
import pandas as pd
import random
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float16)

from biodem.utils_model import read_config, test_n_omics_sf, dataloader_trte


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def rank_feat(path_model_i: str,
              path_pheno_i: str, paths_omics_i: list[str],
              path_rank_o: str,
              random_seeds: list[int], regr_or_clas: bool):
    config = read_config(path_model_i + ".cfg")

    optimal_mdoel = torch.load(path_model_i)

    dataloader_rk, dataloader_, n_class = dataloader_trte(paths_omics_i, path_pheno_i, 0.0, regr_or_clas, 32,1)

    original_loss = test_n_omics_sf(optimal_mdoel, dataloader_rk, config, regr_or_clas)

    each_omics_importance_score = []

    for xom in range(len(paths_omics_i)):
        omics = pd.read_csv(paths_omics_i[xom], index_col=0)
        omics_feature_num = omics.shape[1]
        features_total_score = np.zeros(omics_feature_num)
        features_total_score_df_list = []

        for rand in random_seeds:
            print("----random:{}----".format(rand))

            features_score = []
            OMICS = copy.deepcopy(omics)
            print("-----features_score-----")

            for shuffled_feature in range(omics_feature_num):
                # feature = omics[:, shuffled_feature]
                # np.random.shuffle(feature)

                # !!!
                dataloader_rk, dataloader_, n_class = dataloader_trte(paths_omics_i, path_pheno_i, 0.0, regr_or_clas, 32, 1, rand, xom, shuffled_feature)
                loss = test_n_omics_sf(optimal_mdoel, dataloader_rk, config, regr_or_clas)
                loss = original_loss - loss

                features_score.append(loss)
                omics = copy.deepcopy(OMICS)  # change

                if (shuffled_feature + 1) % 50 == 0:
                    print("-----Calculating the {} feature {} times-----".format((shuffled_feature + 1), rand))

            features_total_score += np.array(features_score)
            features_total_score_df = pd.DataFrame(features_total_score)
            features_total_score_df_list.append(features_total_score_df)

        each_feature_score_by_rand = sum(features_total_score_df_list) / len(features_total_score_df_list)
        each_omics_importance_score.append(each_feature_score_by_rand)

    all_omics_feature_importance = pd.concat(each_omics_importance_score)

    # return all_omics_feature_importance
    all_omics_feature_importance.to_csv(path_rank_o)
