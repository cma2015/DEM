import numpy as np
import pandas as pd
import time
import json
import h5py
import random
import string
import os


def random_string(length: int = 7):
    letters = string.ascii_letters + string.digits
    result = ''.join(random.choice(letters) for _ in range(length))
    return result


def one_hot_encode_snp_matrix(
        snp_matrix: np.ndarray,
        len_one_hot_vec: int = 10,
    ):
    """
    One-hot encode the SNP matrix.
    """
    snp_data = []
    for i_sample in range(snp_matrix.shape[0]):
        snp_vec = snp_matrix[i_sample]
        snp_vec = snp_vec.astype(int)
        snp_vec = np.eye(len_one_hot_vec + 1)[snp_vec][:, 1:]
        snp_vec = snp_vec.reshape(-1)
        snp_data.append(snp_vec)

    return snp_data


def one_hot_encode_phen(
        phen_data: np.ndarray,
        output_dim: int,
    ):
    """
    One-hot encode the phenotype data.
    - If `output_dim` is 1, it means the phenotype is a continuous variable.
    - If `output_dim` is larger than 1, it means the phenotype is a categorical variable. `output_dim` is the number of categories plus 1.
    """
    if output_dim == 1:
        return phen_data
    else:
        # Apply round and one-hot encoding to the phenotype data.
        phen_data = phen_data.round().astype(int)
        phen_data = np.eye(output_dim + 1)[phen_data][:, 0, 1:]
        return phen_data


def workers_start_pts(n_workers: int, n_tasks: int) -> list[int]:
    if n_tasks < 1:
        n_tasks = 1
    if n_workers > n_tasks:
        n_workers = n_tasks
    if n_workers < 1:
        n_workers = 1
    
    n_task_per_worker = n_tasks // n_workers
    task_idx_worker = [i * n_task_per_worker for i in range(n_workers)]
    task_idx_worker.append(n_tasks)
    
    return task_idx_worker



def gen_genes_snps(
        gene_ids: list[str],
        snp_gene_relation: list[list[str]],
    ) -> list[list[int]]:
    """
    Generate `genes_snp: list[list[int]]` for sparse layer initialization
    """
    genes_snp: list[list[int]] = []
    for i_gene in range(len(gene_ids)):
        # Not available: gene_idx = np.where(np.isin([gene_ids_available[i_gene]], snp_gene_relation))[0]
        gene_snps_i: list[int] = []
        for i_snp in range(len(snp_gene_relation)):
            if gene_ids[i_gene] in snp_gene_relation[i_snp]:
                gene_snps_i.append(i_snp)
        genes_snp.append(gene_snps_i)
    return genes_snp


def process_avail_snp(
        path_h5_snp_matrix: str,
        path_json_snp_gene_relation: str,
        return_or_save: bool = False,
    ):
    """
    Read and process available data from a HDF5 of SNP matrix and a JSON of SNP-gene relation.

    Input:
    - `path_h5_snp_matrix`: Path to a HDF5 file that contains a SNP matrix.
    - `path_json_snp_gene_relation`: Path to a JSON file that contains a SNP-gene relation.
        - Each element is a list of gene names that the SNP is in.
        - The length of should be equal to the number of SNPs.
    - `return_or_save`: Whether to return a dictionary of processed data or save them to files.
    """
    # Read a SNP matrix from HDF5 file to a numpy array
    h5file = h5py.File(path_h5_snp_matrix, 'r')
    snp_matrix_h5 = h5file['encoded_matrix'][:]
    snp_ids_h5 = h5file['snp_ids'][:]
    sample_ids_h5 = h5file['sample_ids'][:]
    h5file.close()
    snp_ids_h5 = [s.decode('utf-8') for s in snp_ids_h5]
    sample_ids_h5 = [s.decode('utf-8') for s in sample_ids_h5]

    # Read a SNP-gene relation from JSON file
    with open(path_json_snp_gene_relation, 'r') as f:
        in_json = json.load(f)
    snp_ids_json = in_json['snp_ids']
    # gene_ids_json = in_json['gene_ids']
    snp_gene_relation = in_json['gene_list']

    # Fetch available SNPs and genes from the SNP matrix and SNP-gene relation
    snp_ids_available = np.sort(list(set(snp_ids_h5) & set(snp_ids_json))).tolist()

    # Remove SNPs that are not available in the SNP matrix
    snp_mat_cols2rm = np.where(np.isin(snp_ids_h5, snp_ids_available) == False)[0]
    snp_matrix = np.delete(snp_matrix_h5, snp_mat_cols2rm, axis=1)
    snp_ids_h5 = np.delete(snp_ids_h5, snp_mat_cols2rm)
    
    # Sort snp_ids_h5 and snp_matrix according to snp_ids_h5
    sort_idx = np.argsort(snp_ids_h5)
    snp_ids_h5 = [snp_ids_h5[i] for i in sort_idx]
    snp_matrix = np.take(snp_matrix, sort_idx, axis=1)

    # Sort sample_ids_h5 and snp_matrix according to sample_ids_h5
    sort_idx = np.argsort(sample_ids_h5)
    sample_ids_h5 = [sample_ids_h5[i] for i in sort_idx]
    snp_matrix = np.take(snp_matrix, sort_idx, axis=0)
    
    # Remove SNPs that are not available in the SNP-gene relation
    snp_gene_relat2keep = np.where(np.isin(snp_ids_json, snp_ids_available))[0]
    if len(snp_gene_relat2keep) < len(snp_ids_json):
        snp_gene_relation = snp_gene_relation[snp_gene_relat2keep]
        snp_ids_json = snp_ids_json[snp_gene_relat2keep]
    gene_ids_available = np.sort(np.unique(np.concatenate(snp_gene_relation))).tolist()

    # Sort snp_ids_json and snp_gene_relation according to snp_ids_json
    sort_idx = np.argsort(snp_ids_json)
    snp_ids_json = [snp_ids_json[i] for i in sort_idx]
    snp_gene_relation = [snp_gene_relation[i] for i in sort_idx]

    genes_snps = gen_genes_snps(gene_ids_available, snp_gene_relation)

    if return_or_save:
        return {
            'snp_matrix': snp_matrix,
            'sample_ids': sample_ids_h5,
            'snp_ids': snp_ids_available,
            'genes_snps': genes_snps,
            'gene_ids': gene_ids_available,
            'snp_gene_relation': snp_gene_relation,
        }
    
    else:
        # Save processed data to files, naming with a timestamp in the format of YYYYMMDDHHMMSS
        timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
        path_h5_processed = os.path.join(os.path.dirname(path_h5_snp_matrix), f'processed_data_{timestamp}.h5')

        h5file = h5py.File(path_h5_processed, 'w')
        h5file.create_dataset('snp_matrix', data=snp_matrix)
        h5file.create_dataset('sample_ids', data=[s.encode('utf-8') for s in sample_ids_h5])
        h5file.create_dataset('snp_ids', data=[s.encode('utf-8') for s in snp_ids_available])
        h5file.create_dataset('gene_ids', data=[s.encode('utf-8') for s in gene_ids_available])
        h5file.close()

        # Write genes_snps to a JSON file
        path_json_genes_snps = os.path.join(os.path.dirname(path_h5_snp_matrix), f'processed_data_genes_snps_{timestamp}.json')
        with open(path_json_genes_snps, 'w') as f:
            json.dump(genes_snps, f)
        
        # Write snp_gene_relation to a JSON file
            path_json_processed_snp_gene_rel = os.path.join(os.path.dirname(path_h5_snp_matrix), f'processed_data_snp_gene_relation_{timestamp}.json')
        with open(path_json_processed_snp_gene_rel, 'w') as f:
            json.dump(snp_gene_relation, f)

        return None


def read_processed_data(
        path_h5_processed: str,
        path_json_genes_snps: str | None = None,
    ) -> dict:
    """
    Read processed data from a HDF5 file and a JSON file.
    """
    # Read processed data from HDF5 file
    h5file = h5py.File(path_h5_processed, 'r')
    snp_matrix = h5file['snp_matrix'][:]
    sample_ids = [s.decode('utf-8') for s in h5file['sample_ids'][:]]
    snp_ids = [s.decode('utf-8') for s in h5file['snp_ids'][:]]
    gene_ids = [s.decode('utf-8') for s in h5file['gene_ids'][:]]
    h5file.close()
    
    if path_json_genes_snps is not None:
        # Read genes' SNPs from JSON file
        with open(path_json_genes_snps, 'r') as f_json:
            genes_snps = json.load(f_json)

        return {
            'snp_matrix': snp_matrix,
            'sample_ids': sample_ids,
            'snp_ids': snp_ids,
            'gene_ids': gene_ids,
            'genes_snps': genes_snps,
        }
    
    else:
        return {
            'snp_matrix': snp_matrix,
            'sample_ids': sample_ids,
            'snp_ids': snp_ids,
            'gene_ids': gene_ids,
        }


def read_into_trnval(
        snp_matrix: np.ndarray,
        sample_ids_in_mat: list[str],
        path_csv_pheno_trn: str,
        path_csv_pheno_val: str,
        which_trait: str | None = None,
    ):
    """
    Read and divide data into trn and tst
    """
    phenotypes_trn_df = pd.read_csv(path_csv_pheno_trn, index_col=0)
    phenotypes_val_df = pd.read_csv(path_csv_pheno_val, index_col=0)

    if which_trait is not None:
        phenotypes_trn_df = phenotypes_trn_df[[which_trait]]
        phenotypes_val_df = phenotypes_val_df[[which_trait]]

    phenotypes_trn = phenotypes_trn_df.values
    phenotypes_val = phenotypes_val_df.values
    
    sample_ids_trn = phenotypes_trn_df.index.values.astype(str)
    sample_ids_val = phenotypes_val_df.index.values.astype(str)
    
    sortperm_sample_ids_trn = np.argsort(sample_ids_trn)
    sortperm_sample_ids_val = np.argsort(sample_ids_val)

    phenotypes_trn = np.take(phenotypes_trn, sortperm_sample_ids_trn, axis=0)
    phenotypes_val = np.take(phenotypes_val, sortperm_sample_ids_val, axis=0)

    sample_ids_trn = np.take(sample_ids_trn, sortperm_sample_ids_trn, axis=0)
    sample_ids_val = np.take(sample_ids_val, sortperm_sample_ids_val, axis=0)
    
    indices_samples4trn = np.where(np.isin(sample_ids_in_mat, sample_ids_trn))[0]
    indices_samples4val = np.where(np.isin(sample_ids_in_mat, sample_ids_val))[0]
    
    if len(indices_samples4trn) != len(sample_ids_trn):
        raise ValueError('Some sample IDs in the CSV file are not found in the SNP matrix.')
    if len(indices_samples4val) != len(sample_ids_val):
        raise ValueError('Some sample IDs in the CSV file are not found in the SNP matrix.')
    
    snp_matrix_trn = np.take(snp_matrix, indices_samples4trn, axis=0)
    snp_matrix_val = np.take(snp_matrix, indices_samples4val, axis=0)

    sample_ids_in_mat_trn = np.take(sample_ids_in_mat, indices_samples4trn, axis=0)
    sample_ids_in_mat_val = np.take(sample_ids_in_mat, indices_samples4val, axis=0)

    snp_matrix_trn = np.take(snp_matrix_trn, np.argsort(sample_ids_in_mat_trn), axis=0)
    snp_matrix_val = np.take(snp_matrix_val, np.argsort(sample_ids_in_mat_val), axis=0)

    return phenotypes_trn, phenotypes_val, snp_matrix_trn, snp_matrix_val


def read_into_test(
        snp_matrix: np.ndarray,
        sample_ids_in_mat: list[str],
        path_csv_pheno_tst: str,
        which_trait: str | None = None,
    ):
    """
    Read and pick out test data from the SNP matrix.

    - `path_csv_pheno_tst`: Path to a CSV file that contains the phenotype data for test set, with values standardized independantly.
    """
    phenotypes_tst_df = pd.read_csv(path_csv_pheno_tst, index_col=0)
    if which_trait is not None:
        phenotypes_tst_df = phenotypes_tst_df[[which_trait]]
    phenotypes_tst = phenotypes_tst_df.values
    
    sample_ids_tst = phenotypes_tst_df.index.values.astype(str)
    
    sortperm_sample_ids_tst = np.argsort(sample_ids_tst)
    phenotypes_tst = np.take(phenotypes_tst, sortperm_sample_ids_tst, axis=0)
    sample_ids_tst = np.take(sample_ids_tst, sortperm_sample_ids_tst, axis=0)
    
    indices_samples4tst = np.where(np.isin(sample_ids_in_mat, sample_ids_tst))[0]
    if len(indices_samples4tst) != len(sample_ids_tst):
        print(f'\nThere are {len(sample_ids_tst) - len(indices_samples4tst)} sample IDs in the CSV file that are not found in the SNP matrix.\n')
        print(sample_ids_tst)
        print(sample_ids_in_mat)
        raise ValueError('Some sample IDs in the CSV file are not found in the SNP matrix.')
    
    snp_matrix_tst = np.take(snp_matrix, indices_samples4tst, axis=0)
    sample_ids_in_mat_tst = np.take(sample_ids_in_mat, indices_samples4tst, axis=0)
    snp_matrix_tst = np.take(snp_matrix_tst, np.argsort(sample_ids_in_mat_tst), axis=0)

    return phenotypes_tst, snp_matrix_tst, sample_ids_tst


def gen_ncv_omics_filepaths(
        data_dir: str,
        idx_outer: int | str,
        idx_inner: int | str | None,
        omics_titles: list[str] | None = None,
        f_format = '.csv',
        suffix_outer = 'fold',
        suffix_train = 'tr',
        suffix_validation_or_test = 'te',
    ):
    """
    Generate file paths for training, validation, and testing omics data.
    It also supports label/phenotypic file path generation.

    Note:
    - If `idx_inner` is None, then the file paths for training and validation data are generated for the outer fold.
    - If `idx_inner` is not None, then the file paths for training and validation data are generated for the inner fold.
    - If `omics_titles` is None, then all available omics types in `data_dir` are used.

    Example paths:
    - `data_dir/0fold/0_snp_inner_tr.csv`
    - `data_dir/0fold/0_snp_inner_te.csv`
    - `data_dir/0fold/0_inner_zscore_labels_te.csv`
    """
    if idx_inner is None:
        idx_inner = idx_outer
        dir_outer_fold = data_dir
    else:
        dir_outer_fold = os.path.join(data_dir, str(idx_outer) + suffix_outer)
    
    files_ = sorted(os.listdir(dir_outer_fold))
    if omics_titles is None:
        # Find all available omics types in data_dir
        omics_titles = []
        for file_name in files_:
            if file_name.endswith(f_format):
                omics_title = file_name.split('_')[1]
                if omics_title not in omics_titles:
                    omics_titles.append(omics_title)
    
    paths_trn = []
    paths_val = []
    for file_name in files_:
        if file_name.endswith(f_format):
            sections_file_name = file_name.split('_')
            if sections_file_name[1] in omics_titles:
                if sections_file_name[0] == str(idx_inner):
                    if sections_file_name[-1] == suffix_train + f_format:
                        paths_trn.append(os.path.join(dir_outer_fold, file_name))
                    elif sections_file_name[-1] == suffix_validation_or_test + f_format:
                        paths_val.append(os.path.join(dir_outer_fold, file_name))
    
    if len(paths_trn) == 0 or len(paths_val) == 0:
        raise ValueError(f'No training or validation data found in {dir_outer_fold}')
    return paths_trn, paths_val, omics_titles


def gen_ncv_pheno_filenames(
        which_outer_fold: int | str,
        which_inner_fold: int | str,
        dir_inner_ph: str,
        idx_found = 0,
    ):
    path_csv_pheno_trn, path_csv_pheno_val, _ = gen_ncv_omics_filepaths(dir_inner_ph, which_outer_fold, which_inner_fold)
    # if len(path_csv_pheno_trn) != 1 or len(path_csv_pheno_val) != 1:
    #     raise ValueError('Multiple phenotype files found in the directory.')
    return path_csv_pheno_trn[idx_found], path_csv_pheno_val[idx_found]


def rm_ckpt(ckpt_dir: str, rmALL: bool = False):
    """
    Remove checkpoints from a versions directory.
    """
    if not os.path.isdir(ckpt_dir):
        print("Error: {} is not a directory".format(ckpt_dir))
        exit(1)
    
    version_dirs = os.listdir(ckpt_dir)
    version_ids = [int(v.split("_")[1]) for v in version_dirs]
    # Get sortperm of version_ids
    sortperm = sorted(range(len(version_ids)), key=lambda k: version_ids[k])

    if rmALL:
        version_dirs = [version_dirs[i] for i in sortperm]
        print("Remove all versions")
    else:
        version_dirs = [version_dirs[i] for i in sortperm[:-2]]
        print("Remove versions from {} to {}".format(version_dirs[0], version_dirs[-1]))
    
    for dir_version in version_dirs:
        if os.path.isdir(os.path.join(ckpt_dir, dir_version)):
            for file_name in os.listdir(os.path.join(ckpt_dir, dir_version, "checkpoints")):
                if file_name.endswith(".ckpt"):

                    os.remove(os.path.join(ckpt_dir, dir_version, "checkpoints", file_name))
                    print("Remove {} in {}".format(file_name, os.path.join(ckpt_dir, dir_version, "checkpoints")))

    print("Done!")


def collect_models_paths(dir_models: str) -> dict[str, pd.DataFrame]:
    """
    Collect trained models for each fold in nested cross-validation.
    """
    if os.path.exists(dir_models) == False:
        raise ValueError(f'Directory {dir_models} does not exist.')

    # Find all trained models in the directory `dir_trained_models` and its subdirectories
    trained_models = [os.path.join(dirpath, f)
                    for dirpath, dirnames, files in os.walk(dir_models)
                    for f in files if f.endswith('.ckpt')]
    trained_models.sort()
    print(f'Found {len(trained_models)} trained models\n')

    # Pick ids of outer and inner folds, val_loss and version from ckpt file paths
    val_loss_values = [float(os.path.basename(path_x).split('-')[3].split('=')[1].split('.ckpt')[0]) for path_x in trained_models]
    version_values = [path_x.split('/')[-3] for path_x in trained_models]
    ncv_inner_x = [int(path_x.split('/')[-4].split('_')[-1]) for path_x in trained_models]
    ncv_outer_x = [int(path_x.split('/')[-4].split('_')[-2]) for path_x in trained_models]

    # Create a dataframe with the above values
    df_trained_models = pd.DataFrame(
        {'ncv_outer_x': ncv_outer_x,
        'ncv_inner_x': ncv_inner_x,
        'val_loss': val_loss_values,
        'version': version_values,
        'path_ckpt': trained_models})

    # Pick the best model based on val_loss between the versions of the same outer and inner fold
    df_best_versions = df_trained_models.loc[df_trained_models.groupby(['ncv_outer_x', 'ncv_inner_x'])['val_loss'].idxmin()]

    # Pick the best model based on val_loss between inner folds of the same outer fold
    df_best_inner_folds = df_best_versions.loc[df_best_versions.groupby(['ncv_outer_x'])['val_loss'].idxmin()]

    # return df_best_inner_folds, df_best_versions, df_trained_models
    dict_out = {'best_inner_folds': df_best_inner_folds, 'best_versions': df_best_versions, 'models': df_trained_models}
    
    return dict_out
