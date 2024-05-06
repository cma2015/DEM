"""
biodem
DEM in Python.

"""

name = "biodem"
__version__ = "0.6.0"
__author__ = 'Chenhua Wu, Yanlin Ren'
__credits__ = 'Northwest A&F University'

from .module_data_prep import filter_na_pheno, impute_omics, select_varpca, select_rf
from .module_data_prep_ncv import KFoldSplitter, data_prep_ncv_regre, data_prep_ncv_class
from .utils import gen_ncv_omics_filepaths, gen_ncv_pheno_filenames, process_avail_snp, workers_start_pts, rm_ckpt, collect_models_paths
from .module_snp import SNP2GeneTrain, SNP2GenePipeline, SNPDataModule, snp_to_gene, train_snp2gene
from .module_dem import DEMTrain, DEMTrainPipeline, DEMLTNDataModule, DEMPredict, predict_pheno, DEMFeatureRanking, rank_feat
