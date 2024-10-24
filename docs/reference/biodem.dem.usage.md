# How to use `biodem.dem`

## 1. Nested cross-validation and data preprocessing

Please checkout the documentations at [Modules > Utilities > Preprocessing Data > `OptimizeLitdataNCV`](biodem.utils.data_ncv.md#biodem.utils.data_ncv.OptimizeLitdataNCV).

This is an example of running the module:

``` py title="run_dem_prep.py"
import os
import sys
from biodem import OptimizeLitdataNCV


trait_name = sys.argv[1]
which_o = int(sys.argv[2])
which_i = int(sys.argv[3])

if trait_name.startswith("all"):
    which_trait = None
else:
    which_trait = [trait_name]


if __name__ == "__main__":
    k_outer = 10
    k_inner = 5
    dir_home = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(dir_home, "run_dem", trait_name, "litdata")
    
    path_labels = os.path.join(dir_home, "data_prep", "phenotypes.csv")

    path_transformed_genotypes = os.path.join(dir_home, "run_s2g", trait_name, "transf")
    path_metabolome = os.path.join(dir_home, "data_prep", "omics_metabolome.parquet")
    path_fpkm = os.path.join(dir_home, "data_prep", "omics_fpkm_log2.parquet")
    
    dict_omics = {
        "transcriptome": path_fpkm,
        "metabolome": path_metabolome,
        "genotype": path_transformed_genotypes,
    }

    _opt = OptimizeLitdataNCV(
        paths_omics = dict_omics,
        path_label = path_labels,
        output_dir = output_dir,
        k_outer = k_outer,
        k_inner = k_inner,
        which_outer_inner = [which_o, which_i],
        col2use_in_labels = which_trait,
    )
    _opt.run_optimization()
```

## 2. Dual-extraction modeling

Please checkout the documentations at [Modules > DEM > Pipeline > `DEMFitPipe`](biodem.dem.pipeline.md#biodem.dem.pipeline.DEMFitPipe).

This is an example of running the module:

``` py title="run_dem_fit.py"
import os
import sys
from biodem import DEMFitPipe


if len(sys.argv) < 2:
    raise ValueError('Please specify the TRAIT NAME')
trait_name = sys.argv[1]

if len(sys.argv) < 4:
    print('Start default NCV: 10 outer folds and 5 inner folds')
    list_ncv = [[i,j] for i in range(10) for j in range(5)]
else:
    print('Start with NCV: {} outer folds and {} inner folds'.format(sys.argv[2], sys.argv[3]))
    list_ncv = [[int(sys.argv[2]), int(sys.argv[3])]]

work_dir_home = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_dem", trait_name)
litdata_dir = os.path.join(work_dir_home, 'litdata')
is_regression = True
log_dir = os.path.join(work_dir_home, 'models')


if __name__ == '__main__':
    _pipe = DEMFitPipe(
        litdata_dir=litdata_dir,
        list_ncv=list_ncv,
        log_dir=log_dir,
        regression=is_regression,
    )
    _pipe.train_pipeline()
```

## 3. Feature ranking

Please checkout the documentations at [Modules > DEM > Feature ranking > `DEMFeatureRanking`](biodem.dem.rank.md#biodem.dem.rank.DEMFeatureRanking).

This is an example of running the module:

``` py title="run_dem_rank.py"
import os
import sys
from biodem import DEMFeatureRanking


if len(sys.argv) < 3:
    raise ValueError('Please input trait name and outer index')
trait_name = sys.argv[1]
which_outer = int(sys.argv[2])

work_dir_home = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_dem", trait_name)
litdata_dir = os.path.join(work_dir_home, 'litdata')
log_dir = os.path.join(work_dir_home, 'models')

rank_result_path = os.path.join(work_dir_home, "feature_rank", f"rank_result_{trait_name}_outer+{which_outer}.csv")
random_seeds = [1000+i for i in range(20)]


if __name__ == "__main__":
    _feat_rank = DEMFeatureRanking()
    _feat_rank.run_a_outer(
        ncv_litdata_dir = litdata_dir,
        fit_log_dir = log_dir,
        which_outer = which_outer,
        output_path = rank_result_path,
        random_states = random_seeds,
    )
    
    # Collect ranks if several outer testset ranking results already exist
    # _feat_rank.collect_ranks(os.path.dirname(rank_result_path), os.path.join(os.path.dirname(rank_result_path), "rank_merged_sorted.csv"))
```
