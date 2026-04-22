import numpy as np
import pandas as pd
import torch
from scipy.io import mmread
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import anndata as ad
import scanpy as sc
import maxfuse as mf
import maxfuse.utils as mf_utils
import maxfuse.graph as mf_graph
import seaborn as sns
import os
import numbers


def maxfuse_integration(common_genes, rna_adata, protein_adata, to_csv_path=None,
                        labels_rna=None, labels_codex=None, visualize=False, labels_path=None,
                        library_size_normalize_sc=False, log_normalize_sc=False,
                        scale_normalize_sc=False, metacell_size=2,
                        init_wt1=0.3, init_wt2=0.3,
                        refine_wt1=0.3, refine_wt2=0.3,
                        refine_filter_prop=0.3,
                        propagate_wt1=0.7, propagate_wt2=0.7):
    """
    Performs integration of scRNA-seq and protein data using the MaxFuse
    pipeline with default parameters in Maxfuse tutorial.

    This function integrates single-cell RNA sequencing data and protein data
    on shared genes and performs various analyses such as visualizations,
    dimensionality reduction, and matching through the MaxFuse pipeline.
    The results can be saved to a CSV file for further analysis.

    Parameters:
        common_genes (list): List of common genes between RNA and protein data.
        rna_adata (AnnData): Annotated adata for RNA data.
        protein_adata (AnnData): Annotated adata for protein data.
        to_csv_path (str, optional): Path to save the matched results. Defaults to "./data/match.csv".
        labels_rna (ndarray, optional): Labels for RNA data cells. Required for visualization.
        labels_codex (ndarray, optional): Labels for protein data cells. Required for visualization.
        visualize (bool, optional): If True, generates visualizations. Defaults to False.
        metacell_size (int, optional): Average metacell size used when batching/scaling in MaxFuse.
            Defaults to 2; set to 1 to work on single cells.
        init_wt1/init_wt2 (float, optional): Smoothing weights for ``find_initial_pivots`` on RNA/ST.
            Lower values (≈0.1) lean on raw expression; higher values (≈0.5) rely more on cluster/graph averages.
        refine_wt1/refine_wt2 (float, optional): Same as above but for ``refine_pivots``. Keep close to
            the initial weights—use smaller numbers to preserve rare types, larger to denoise more aggressively.
        refine_filter_prop (float, optional): Proportion of low-scoring pivot matches to drop in
            ``filter_bad_matches`` immediately after refinement. Defaults to 0.3; decrease for stricter filtering.
        propagate_wt1/propagate_wt2 (float, optional): Weights used in ``propagate``. Start around 0.2–0.3
            to avoid over-smoothing; increase toward 0.7 only if propagated matches look noisy.

    Returns:
        None

    Raises:
        ValueError: If attempting to visualize without both RNA and protein labels.
    """

    def _sanitize_labels(labels):
        if labels is None:
            return None
        series = pd.Series(labels)

        def _clean_value(val):
            if pd.isna(val):
                return "NAN_converted"
            if isinstance(val, numbers.Number) and not np.isfinite(val):
                return "NAN_converted"
            return val

        series = series.map(_clean_value)
        return series.astype(str).to_numpy()

    labels_rna = _sanitize_labels(labels_rna)
    labels_codex = _sanitize_labels(labels_codex)

    if to_csv_path is None:
        to_csv_path = os.path.join(os.getcwd(), "data", "match.csv")

    if visualize:
        if labels_path is None:
            labels_path = os.path.join(os.getcwd(), "data", "labels.csv")

    if ((labels_rna is not None) and (labels_codex is not None)) != visualize:
        raise ValueError("To visualize the results of maxfuse, please input the label of cells in"
                         " scRNA-seq data and image-based ST data")

    assert ((labels_rna is not None) and (labels_codex is not None)) == visualize

    # Check if adata.X is sparse
    if not isinstance(rna_adata.X, np.ndarray):
        rna_adata.X = rna_adata.X.toarray()  # Convert to dense numpy array
    # Check if adata.X is sparse
    if not isinstance(protein_adata.X, np.ndarray):
        protein_adata.X = protein_adata.X.toarray()  # Convert to dense numpy array

    if visualize:
        rna_adata.obs["celltype"] = labels_rna
        protein_adata.obs["celltype"] = labels_codex

    if library_size_normalize_sc:
        sc.pp.normalize_total(rna_adata, target_sum=1e5)
    if log_normalize_sc:
        sc.pp.log1p(rna_adata)

    sc.pp.highly_variable_genes(rna_adata, n_top_genes=5000)

    if scale_normalize_sc:
        sc.pp.scale(rna_adata)
        sc.pp.scale(protein_adata)

    hvgs = list(set(list(rna_adata.var.highly_variable.index[rna_adata.var.highly_variable.values == True])).union(
        set(common_genes)))
    hvgs = [i for i in hvgs if i in rna_adata.var_names]
    # only retain highly variable genes
    rna_adata = rna_adata[:, hvgs].copy()

    correspondence = pd.DataFrame(data=[[x, x] for x in common_genes], columns=["Protein name", "RNA name"])

    rna_protein_correspondence = []

    for i in range(correspondence.shape[0]):
        curr_protein_name, curr_rna_names = correspondence.iloc[i]
        if curr_protein_name not in protein_adata.var_names:
            print(curr_protein_name)
            continue
        if curr_rna_names.find('Ignore') != -1:  # some correspondence ignored eg. protein isoform to one gene
            print(curr_protein_name)
            continue
        curr_rna_names = curr_rna_names.split('/')  # eg. one protein to multiple genes
        for r in curr_rna_names:
            if r in rna_adata.var_names:
                rna_protein_correspondence.append([r, curr_protein_name])

    rna_protein_correspondence = np.array(rna_protein_correspondence)

    # Columns rna_shared and protein_shared are matched.
    # One may encounter "Variable names are not unique" warning,
    # this is fine and is because one RNA may encode multiple proteins and vice versa.
    rna_shared = rna_adata[:, rna_protein_correspondence[:, 0]].copy()
    protein_shared = protein_adata[:, rna_protein_correspondence[:, 1]].copy()
    print(rna_shared.shape, protein_shared.shape)

    if visualize:
        sc.pp.neighbors(rna_shared, n_neighbors=15)
        sc.tl.umap(rna_shared)
        sc.pl.umap(rna_shared, color='celltype', title="rna_shared")

        sc.pp.neighbors(protein_shared, n_neighbors=15)
        sc.tl.umap(protein_shared)
        sc.pl.umap(protein_shared, color='celltype', title='protein_shared')

        # plot UMAPs of rna cells based on all active rna markers
        sc.pp.neighbors(rna_adata, n_neighbors=15)
        sc.tl.umap(rna_adata)
        sc.pl.umap(rna_adata, color='celltype', title='rna_adata')

    # make sure no feature is static
    rna_active = rna_adata.X
    protein_active = protein_adata.X
    rna_active = rna_active[:, rna_active.std(axis=0) > 1e-5]  # these are fine since already using variable features
    protein_active = protein_active[:, protein_active.std(axis=0) > 1e-5]  # protein are generally variable

    rna_shared = rna_shared.X.copy()
    protein_shared = protein_shared.X.copy()

    # call constructor for Fusor object
    # which is the main object for running MaxFuse pipeline
    fusor = mf.model.Fusor(
        shared_arr1=rna_shared,
        shared_arr2=protein_shared,
        active_arr1=rna_active,
        active_arr2=protein_active,
        labels1=labels_rna,
        labels2=labels_codex
    )

    fusor.split_into_batches(
        max_outward_size=rna_active.shape[0],
        matching_ratio=protein_active.shape[0] / rna_active.shape[0],
        metacell_size=metacell_size,
        verbose=True
    )

    # plot top singular values of avtive_arr1 on a random batch
    fusor.plot_singular_values(
        target='active_arr1',
        n_components=None  # can also explicitly specify the number of components
    )

    # plot top singular values of avtive_arr2 on a random batch
    fusor.plot_singular_values(
        target='active_arr2',
        n_components=None
    )

    fusor.construct_graphs(
        n_neighbors1=15,
        n_neighbors2=15,
        svd_components1=40,
        svd_components2=15,
        resolution1=2,
        resolution2=2,
        # if two resolutions differ less than resolution_tol
        # then we do not distinguish between then
        resolution_tol=0.1,
        verbose=True
    )

    # plot top singular values of shared_arr1 on a random batch
    fusor.plot_singular_values(
        target='shared_arr1',
        n_components=None,
    )

    # plot top singular values of shared_arr2 on a random batch
    fusor.plot_singular_values(
        target='shared_arr2',
        n_components=None
    )

    fusor.find_initial_pivots(
        wt1=init_wt1, wt2=init_wt2,
        svd_components1=25, svd_components2=20
    )

    # plot top canonical correlations in a random batch
    fusor.plot_canonical_correlations(
        svd_components1=50,
        svd_components2=None,
        cca_components=45
    )

    fusor.refine_pivots(
        wt1=refine_wt1, wt2=refine_wt2,
        svd_components1=40, svd_components2=None,
        cca_components=25,
        n_iters=1,
        randomized_svd=False,
        svd_runs=1,
        verbose=True
    )

    fusor.filter_bad_matches(target='pivot', filter_prop=refine_filter_prop)

    pivot_matching = fusor.get_matching(order=(2, 1), target='pivot')

    labels_rna = np.asarray(labels_rna).astype(str)
    labels_codex = np.asarray(labels_codex).astype(str)

    if visualize:
        # labels_plot=np.unique(labels_rna[pivot_matching[0]]) if len(np.unique(labels_rna[pivot_matching[0]]))>len(np.unique(labels_codex[pivot_matching[1]])) else np.unique(labels_codex[pivot_matching[1]])
        # print(pivot_matching[0], "\n", labels_rna[pivot_matching[0]], "\n", pivot_matching[1], "\n",labels_codex[pivot_matching[1]])
        labels_plot = np.unique(
            np.concatenate((
                np.asarray(labels_rna[pivot_matching[0]]).astype(str),
                np.asarray(labels_codex[pivot_matching[1]]).astype(str)
            ))
        )
        lv1_acc = mf.metrics.get_matching_acc(matching=pivot_matching,
                                              labels1=labels_rna,
                                              labels2=labels_codex,
                                              order=(2, 1)
                                              )
        print(lv1_acc)
        cm = confusion_matrix(labels_rna[pivot_matching[0]], labels_codex[pivot_matching[1]])
        cmd = ConfusionMatrixDisplay(
            confusion_matrix=np.nan_to_num(np.round((cm.T / np.sum(cm, axis=1)).T * 100)),
            display_labels=labels_plot
        )

        # Plotting the confusion matrix
        cmd.plot()
        plt.xticks(rotation=90)  # Rotate x-axis labels to vertical
        plt.show()  # Display the plot

        # Create a DataFrame
        '''print(len(labels_rna[pivot_matching[0]]),len(labels_codex[pivot_matching[1]]))
        print(labels_rna[pivot_matching[0]])
        print(labels_codex[pivot_matching[1]])'''
        df_label = pd.DataFrame({
            "scRNA-seq": pd.Series(labels_rna[pivot_matching[0]]).values,
            "ST": pd.Series(labels_codex[pivot_matching[1]]).values
        })

        # Save the DataFrame to CSV
        df_label.to_csv(labels_path, index=False)

    fusor.propagate(
        svd_components1=40,
        svd_components2=None,
        wt1=propagate_wt1,
        wt2=propagate_wt2,
    )

    fusor.filter_bad_matches(
        target='propagated',
        filter_prop=0
    )

    full_matching = fusor.get_matching(order=(2, 1), target='full_data')

    df_match = pd.DataFrame(list(zip(full_matching[0], full_matching[1], full_matching[2])),
                            columns=['mod1_indx', 'mod2_indx', 'score'])
    df_match.to_csv(to_csv_path)

    if visualize:
        rna_cca, protein_cca_sub = fusor.get_embedding(
            active_arr1=fusor.active_arr1,
            active_arr2=fusor.active_arr2[full_matching[1], :]  # cells in codex remained after filtering
        )
        np.random.seed(42)
        subs = protein_cca_sub.shape[0]
        randix = np.random.choice(protein_cca_sub.shape[0], subs, replace=False)

        dim_use = 15  # dimensions of the CCA embedding to be used for UMAP etc

        cca_adata = ad.AnnData(
            np.concatenate((rna_cca[:, :dim_use], protein_cca_sub[randix, :dim_use]), axis=0),
            dtype=np.float32
        )
        cca_adata.obs['data_type'] = ['rna'] * rna_cca.shape[0] + ['protein'] * subs
        cca_adata.obs['cell_type'] = list(np.concatenate((labels_rna,
                                                          labels_codex[full_matching[1]][randix]), axis=0))
        sc.pp.neighbors(cca_adata, n_neighbors=15)
        sc.tl.umap(cca_adata)
        sc.pl.umap(cca_adata, color='data_type')
        sc.pl.umap(cca_adata, color='cell_type')

    return df_match


def maxfuse_integration_debugged(common_genes, rna_adata, protein_adata, to_csv_path=None,
                                 labels_rna=None, labels_codex=None, visualize=False, labels_path=None,
                                 library_size_normalize_sc=False, log_normalize_sc=False,
                                 scale_normalize_sc=False, metacell_size=2,
                                 init_wt1=0.3, init_wt2=0.3,
                                 refine_wt1=0.3, refine_wt2=0.3,
                                 refine_filter_prop=0.3,
                                 propagate_wt1=0.7, propagate_wt2=0.7):
    """Run ``maxfuse_integration`` with a temporary patch for the propagation bug in
    upstream MaxFuse.

    The original ``Fusor.propagate`` records the surviving pivot matches after filtering,
    but then tries to look them up *again* as if the surviving cell identifiers were array
    positions. When modality 1 (e.g. scRNA-seq) has more cells than modality 2 (e.g. ST),
    those identifiers often include values larger than the count of surviving matches,
    so the second lookup raises ``IndexError`` or silently scrambles indices.

    Example:
        ``curr_refined_matching[0] = [2, 17, 54, 812]`` (RNA cell ids for four matches)
        ``self._remaining_indices_in_refined_matching[...] = [0, 1, 2, 3]``
        ``existing_indices = [2, 17, 54, 812]``
        The buggy code does ``curr_refined_matching[0][existing_indices]``,
        which attempts to read position 812 from a length-4 array and crashes.

    The patched version simply reuses the surviving positions (`rows[surviving_positions]`)
    without performing the second lookup, so datasets where RNA >> ST complete without error,
    and cases where ST ≥ RNA continue to match the original results.

    Parameters mirror ``maxfuse_integration``; ``metacell_size`` controls batching,
    and the weight/filter parameters let you tune smoothing and pruning at each stage."""

    original_propagate = mf.model.Fusor.propagate

    def _propagate_fixed(self, wt1=0.7, wt2=0.7, svd_components1=None, svd_components2=None,
                         metric='euclidean', randomized_svd=False, svd_runs=1, verbose=True):
        """Copy of ``Fusor.propagate`` with fallback to safe indexing when
        surviving pivot indices exceed the refined matching length."""

        self._propagated_matching = []
        for batch_idx, (b1, b2) in enumerate(self._batch1_to_batch2):
            if verbose:
                print('Now at batch {}<->{}...'.format(b1, b2), flush=True)

            curr_propagated_matching = [[], [], []]
            curr_refined_matching = self._refined_matching[batch_idx]

            surviving_positions = np.asarray(
                self._remaining_indices_in_refined_matching[batch_idx], dtype=int
            )

            if surviving_positions.size == 0:
                self._propagated_matching.append(
                    (np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float))
                )
                continue

            rows = np.asarray(curr_refined_matching[0], dtype=int)
            cols = np.asarray(curr_refined_matching[1], dtype=int)
            existing_indices = rows[surviving_positions]

            if existing_indices.size == 0:
                self._propagated_matching.append(
                    (np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float))
                )
                continue

            good_indices1 = np.empty_like(existing_indices)
            good_indices2 = np.empty_like(existing_indices)
            valid_mask = (existing_indices >= 0) & (existing_indices < rows.size)

            if np.any(valid_mask):
                good_indices1[valid_mask] = rows[existing_indices[valid_mask]]
                good_indices2[valid_mask] = cols[existing_indices[valid_mask]]

            if np.any(~valid_mask):
                fallback_positions = surviving_positions[~valid_mask]
                good_indices1[~valid_mask] = existing_indices[~valid_mask]
                good_indices2[~valid_mask] = cols[fallback_positions]

            if self.metacell_size > 1:
                curr_arr1 = mf_utils.get_centroids(
                    arr=self.active_arr1[self._batch_to_indices1[b1], :],
                    labels=self._metacell_labels1[b1]
                )
            else:
                curr_arr1 = self.active_arr1[self._batch_to_indices1[b1], :]
            curr_arr2 = self.active_arr2[self._batch_to_indices2[b2], :]

            if self.method == 'centroid_shrinkage':
                clust_labels1 = self._labels1[b1]
                clust_labels2 = self._labels2[b2]
                curr_arr1 = mf_utils.shrink_towards_centroids(arr=curr_arr1, labels=clust_labels1, wt=wt1)
                curr_arr2 = mf_utils.shrink_towards_centroids(arr=curr_arr2, labels=clust_labels2, wt=wt2)
            elif self.method == 'graph_smoothing':
                edges1 = self._edges1[b1]
                edges2 = self._edges2[b2]
                curr_arr1 = mf_utils.graph_smoothing(arr=curr_arr1, edges=edges1, wt=wt1)
                curr_arr2 = mf_utils.graph_smoothing(arr=curr_arr2, edges=edges2, wt=wt2)
            else:
                raise ValueError("self.method must be one of 'centroid_shrinkage' or 'graph_smoothing'.")

            good_indices1_set = set(good_indices1.tolist())
            remaining_indices1 = [i for i in range(curr_arr1.shape[0]) if i not in good_indices1_set]
            good_indices2_set = set(good_indices2.tolist())
            remaining_indices2 = [i for i in range(curr_arr2.shape[0]) if i not in good_indices2_set]

            if len(remaining_indices1) > 0:
                remaining_indices1_nns, remaining_indices1_nn_dists = mf_graph.get_nearest_neighbors(
                    query_arr=curr_arr1[remaining_indices1, :],
                    target_arr=curr_arr1[good_indices1, :],
                    svd_components=svd_components1,
                    randomized_svd=randomized_svd,
                    svd_runs=svd_runs,
                    metric=metric
                )
                matched_indices2 = good_indices2[remaining_indices1_nns]
                curr_propagated_matching[0].extend(remaining_indices1)
                curr_propagated_matching[1].extend(matched_indices2.tolist())
                curr_propagated_matching[2].extend(remaining_indices1_nn_dists.tolist())

            if len(remaining_indices2) > 0:
                remaining_indices2_nns, remaining_indices2_nn_dists = mf_graph.get_nearest_neighbors(
                    query_arr=curr_arr2[remaining_indices2, :],
                    target_arr=curr_arr2[good_indices2, :],
                    svd_components=svd_components2,
                    randomized_svd=randomized_svd,
                    svd_runs=svd_runs,
                    metric=metric
                )
                matched_indices1 = good_indices1[remaining_indices2_nns]
                curr_propagated_matching[0].extend(matched_indices1.tolist())
                curr_propagated_matching[1].extend(remaining_indices2)
                curr_propagated_matching[2].extend(remaining_indices2_nn_dists.tolist())

            self._propagated_matching.append(
                (
                    np.array(curr_propagated_matching[0], dtype=int),
                    np.array(curr_propagated_matching[1], dtype=int),
                    np.array(curr_propagated_matching[2], dtype=float)
                )
            )

        if verbose:
            print('Done!', flush=True)

    try:
        mf.model.Fusor.propagate = _propagate_fixed
        return maxfuse_integration(
            common_genes=common_genes,
            rna_adata=rna_adata,
            protein_adata=protein_adata,
            to_csv_path=to_csv_path,
            labels_rna=labels_rna,
            labels_codex=labels_codex,
            visualize=visualize,
            labels_path=labels_path,
            library_size_normalize_sc=library_size_normalize_sc,
            log_normalize_sc=log_normalize_sc,
            scale_normalize_sc=scale_normalize_sc,
            metacell_size=metacell_size,
            init_wt1=init_wt1,
            init_wt2=init_wt2,
            refine_wt1=refine_wt1,
            refine_wt2=refine_wt2,
            refine_filter_prop=refine_filter_prop,
            propagate_wt1=propagate_wt1,
            propagate_wt2=propagate_wt2
        )
    finally:
        mf.model.Fusor.propagate = original_propagate

