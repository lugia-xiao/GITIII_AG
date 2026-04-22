from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

from gitiii_ag.run_maxfuse import maxfuse_integration, maxfuse_integration_debugged


def _to_dense(x):
    """Convert sparse matrix-like inputs to a NumPy array."""
    if isinstance(x, np.ndarray):
        return x
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)



def _validate_match_df(match_df: pd.DataFrame) -> None:
    required_cols = {"mod1_indx", "mod2_indx"}
    missing = required_cols.difference(match_df.columns)
    if missing:
        raise ValueError(
            f"match_df is missing required columns: {sorted(missing)}. "
            "Expected columns include 'mod1_indx' and 'mod2_indx'."
        )



def _run_check_plots(
    adata: ad.AnnData,
    st_adata: ad.AnnData,
    sc_adata: ad.AnnData,
    match_df: pd.DataFrame,
    all_genes: Sequence[str],
) -> None:
    if "celltype" in adata.obs.columns:
        cell_type_key = "celltype"
    elif "subclass" in adata.obs.columns:
        cell_type_key = "subclass"
    else:
        raise ValueError(
            "check=True requires adata.obs (copied from st_adata.obs) to contain "
            "either 'celltype' or 'subclass'."
        )

    matched_st = list(np.unique(match_df["mod2_indx"].astype(int)))
    shared_genes = [g for g in all_genes if g in sc_adata.var_names]

    adata_imputed = adata[matched_st, shared_genes].copy()
    if adata_imputed.n_vars == 0:
        raise ValueError("No shared genes available for imputed-data QC plots.")
    sc.pp.highly_variable_genes(adata_imputed, n_top_genes=min(2000, adata_imputed.n_vars), subset=True)
    sc.pp.pca(adata_imputed)
    sc.pp.neighbors(adata_imputed, n_neighbors=min(40, max(2, adata_imputed.n_obs - 1)))
    sc.tl.umap(adata_imputed)
    sc.pl.umap(
        adata_imputed,
        color=cell_type_key,
        title="Imputed gene by cell type",
        show=True,
    )

    adata_original = adata[:, [f"{g}_st" for g in st_adata.var_names]].copy()
    sc.pp.pca(adata_original)
    sc.pp.neighbors(adata_original)
    sc.tl.umap(adata_original)
    sc.pl.umap(
        adata_original,
        color=cell_type_key,
        title="Original ST gene by cell type",
        show=True,
    )

    intersection_genes = list(set(st_adata.var_names).intersection(set(sc_adata.var_names)))
    if len(intersection_genes) == 0:
        raise ValueError("No overlapping genes between st_adata.var_names and sc_adata.var_names.")

    data_sc = _to_dense(adata[:, intersection_genes].X).copy()
    data_st = _to_dense(adata[:, [f"{g}_st" for g in intersection_genes]].X).copy()

    if data_sc.shape != data_st.shape:
        raise ValueError(
            f"Shape mismatch between imputed and original overlap matrices: {data_sc.shape} vs {data_st.shape}."
        )

    n_genes = data_sc.shape[1]
    pcc_list = np.empty(n_genes)
    scc_list = np.empty(n_genes)

    for j in range(n_genes):
        x = data_sc[:, j]
        y = data_st[:, j]
        pcc_list[j], _ = pearsonr(x, y)
        scc_list[j], _ = spearmanr(x, y)

    plt.figure(figsize=(7, 5))
    plt.hist(pcc_list, bins=20, edgecolor="black")
    plt.xlabel("Pearson correlation coefficient")
    plt.ylabel("Number of genes")
    plt.title("Distribution of per-gene PCC between data_sc and data_st")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.hist(scc_list, bins=20, edgecolor="black")
    plt.xlabel("Spearman correlation coefficient")
    plt.ylabel("Number of genes")
    plt.title("Distribution of per-gene SCC between data_sc and data_st")
    plt.tight_layout()
    plt.show()



def maxfuse_impute_and_save(
    sc_adata: ad.AnnData,
    st_adata: ad.AnnData,
    st_genes: Sequence[str]=None,
    sc_label=None,
    st_label=None,
    common_genes: Optional[Sequence[str]] = None,
    output_h5ad_path: str = "./adata_imputed.h5ad",
    to_csv_path: Optional[str] = None,
    visualize: bool = False,
    label_path: Optional[str] = None,
    library_size_normalize_sc: bool = False,
    log_normalize_sc: bool = False,
    scale_normalize_sc: bool = False,
    use_my_debugged_function_of_wrong_maxfuse_code: bool = True,
    metacell_size: int = 2,
    init_wt1: float = 0.3,
    init_wt2: float = 0.3,
    refine_wt1: float = 0.3,
    refine_wt2: float = 0.3,
    propagate_wt1: float = 0.7,
    propagate_wt2: float = 0.7,
    refine_filter_prop: float = 0.1,
    check: bool = False,
) -> Tuple[ad.AnnData, pd.DataFrame]:
    """
    Run MaxFuse-based imputation and save the full imputed AnnData object.

    Parameters
    ----------
    sc_adata
        Single-cell RNA AnnData.
    st_adata
        Spatial/transcriptomics/protein-side AnnData to impute onto.
    st_genes
        Final target gene ordering for the imputed output.
    sc_label, st_label
        Labels passed into the MaxFuse integration function.
    common_genes
        Genes shared across modalities for MaxFuse matching. If None, uses the
        intersection of sc_adata.var_names and st_adata.var_names.
    output_h5ad_path
        Path to save the final imputed AnnData object.
    to_csv_path
        Path to save the MaxFuse match table. Defaults to ./data/match.csv.
    visualize, label_path, library_size_normalize_sc, log_normalize_sc,
    scale_normalize_sc, use_my_debugged_function_of_wrong_maxfuse_code,
    metacell_size, init_wt1, init_wt2, refine_wt1, refine_wt2,
    propagate_wt1, propagate_wt2, refine_filter_prop, check
        Passed through from the original implementation.

    Returns
    -------
    adata_imputed, match_df
        The saved full AnnData object and the MaxFuse match dataframe.
    """
    all_genes=st_genes
    if all_genes is None:
        all_genes = sc_adata.var_names.tolist()

    if common_genes is None:
        common_genes = list(set(sc_adata.var_names.tolist()).intersection(set(st_adata.var_names.tolist())))

    if len(common_genes) == 0:
        raise ValueError("No common genes found between sc_adata.var_names and st_adata.var_names.")

    if to_csv_path is None:
        to_csv_path = os.path.join(os.getcwd(), "data", "match.csv")

    output_dir = os.path.dirname(os.path.abspath(output_h5ad_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    csv_dir = os.path.dirname(os.path.abspath(to_csv_path))
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    sc_work = sc_adata.copy()
    st_work = st_adata.copy()

    maxfuse_fn = (
        maxfuse_integration_debugged
        if use_my_debugged_function_of_wrong_maxfuse_code
        else maxfuse_integration
    )

    match_df = maxfuse_fn(
        common_genes=common_genes,
        rna_adata=sc_work,
        protein_adata=st_work,
        to_csv_path=to_csv_path,
        labels_rna=sc_label,
        labels_codex=st_label,
        visualize=visualize,
        labels_path=label_path,
        library_size_normalize_sc=library_size_normalize_sc,
        log_normalize_sc=log_normalize_sc,
        scale_normalize_sc=scale_normalize_sc,
        metacell_size=metacell_size,
        init_wt1=init_wt1,
        init_wt2=init_wt2,
        refine_wt1=refine_wt1,
        refine_wt2=refine_wt2,
        propagate_wt1=propagate_wt1,
        propagate_wt2=propagate_wt2,
        refine_filter_prop=refine_filter_prop,
    )

    _validate_match_df(match_df)

    sc_subset_genes = [g for g in all_genes if g in sc_adata.var_names.tolist()]
    st_only_genes = [g for g in all_genes if g not in sc_adata.var_names.tolist()]

    sc_subset = sc_adata[:, sc_subset_genes].copy()
    sc_subset_X = _to_dense(sc_subset.X)

    st_only_X = _to_dense(st_adata[:, st_only_genes].X) if len(st_only_genes) > 0 else np.zeros((st_adata.n_obs, 0))
    st_full_X = _to_dense(st_adata.X)

    data_matrix = np.zeros((st_adata.shape[0], sc_subset_X.shape[1]), dtype=sc_subset_X.dtype)
    for idx in match_df.index:
        mod2_idx = int(match_df.loc[idx, "mod2_indx"])
        mod1_idx = int(match_df.loc[idx, "mod1_indx"])
        data_matrix[mod2_idx, :] = sc_subset_X[mod1_idx, :]

    matched_st = np.unique(match_df["mod2_indx"].astype(int))
    have_match_flag = np.array([i in matched_st for i in range(st_adata.shape[0])], dtype=bool)

    all_data_matrix = np.concatenate([st_only_X, data_matrix], axis=1)
    all_data_matrix = np.concatenate([all_data_matrix, st_full_X], axis=1)

    var_names = (
        st_only_genes
        + sc_subset_genes
        + [f"{g}_st" for g in st_adata.var_names.tolist()]
    )

    adata_imputed = ad.AnnData(
        X=all_data_matrix,
        obs=st_adata.obs.copy(),
        var=pd.DataFrame(index=var_names),
    )
    adata_imputed.obs["have_match"] = have_match_flag
    adata_imputed.uns["maxfuse_match_df"] = match_df.copy()
    adata_imputed.uns["maxfuse_common_genes"] = list(common_genes)
    adata_imputed.uns["maxfuse_all_genes"] = list(all_genes)

    print(f"Have match {int(np.sum(have_match_flag))}/{adata_imputed.shape[0]}")

    if check:
        _run_check_plots(
            adata=adata_imputed,
            st_adata=st_adata,
            sc_adata=sc_subset,
            match_df=match_df,
            all_genes=all_genes,
        )

    adata_imputed.write(output_h5ad_path)
    print(f"Saved imputed AnnData to: {output_h5ad_path}")

    return adata_imputed, match_df
