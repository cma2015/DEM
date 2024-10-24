"""
**`biodem`**

Please cite our paper if you use this package:

[Dual-extraction modeling: A multi-modal deep-learning architecture for phenotypic prediction and functional gene mining of complex traits](https://doi.org/10.1016/j.xplc.2024.101002)

"""

name = "biodem"
__version__ = "0.9.1"
__author__ = 'Chenhua Wu, Yanlin Ren'
__credits__ = 'Northwest A&F University'

from biodem.utils.uni import read_pkl_gv, train_model, CollectFitLog, get_avail_cpu_count, get_avail_nvgpu, get_map_location
from biodem.utils.data_ncv import OptimizeLitdataNCV, optimize_data_external, DEMDataset, DEMDataModule4Train, DEMDataModule4Uni
from biodem.s2g.pipeline import SNP2GBFitPipe, SNP2GBTransPipe
from biodem.dem.pipeline import DEMFitPipe, DEMPredict
from biodem.dem.rank import DEMFeatureRanking

from biodem import constants
from biodem import utils
from biodem import dem
from biodem import s2g
