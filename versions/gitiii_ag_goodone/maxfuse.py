import numpy as np
import pandas as pd
import torch
from scipy.io import mmread
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import anndata as ad
import scanpy as sc
import maxfuse as mf
import seaborn as sns
import os

def maxfuse_integration(common_genes,rna_adata,protein_adata,to_csv_path=None,
                        labels_rna=None,labels_codex=None,visualize=False,labels_path=None):
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

    Returns:
        None

    Raises:
        ValueError: If attempting to visualize without both RNA and protein labels.
    """
    if to_csv_path is None:
        to_csv_path=os.path.join(os.getcwd(),"data","match.csv")

    if visualize:
        if labels_path is None:
            labels_path=os.path.join(os.getcwd(),"data","labels.csv")

    if ((labels_rna is not None) and (labels_codex is not None))!=visualize:
        raise ValueError("To visualize the results of maxfuse, please input the label of cells in"
                         " scRNA-seq data and image-based ST data")

    assert ((labels_rna is not None) and (labels_codex is not None))==visualize

    # Check if adata.X is sparse
    if not isinstance(rna_adata.X, np.ndarray):
        rna_adata.X = rna_adata.X.toarray()  # Convert to dense numpy array
    # Check if adata.X is sparse
    if not isinstance(protein_adata.X, np.ndarray):
        protein_adata.X = protein_adata.X.toarray()  # Convert to dense numpy array

    if visualize:
        rna_adata.obs["celltype"]=labels_rna
        protein_adata.obs["celltype"]=labels_codex

    sc.pp.highly_variable_genes(rna_adata, n_top_genes=5000)
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
        labels1=None,
        labels2=None
    )

    fusor.split_into_batches(
        max_outward_size=rna_active.shape[0],
        matching_ratio=protein_active.shape[0] / rna_active.shape[0],
        metacell_size=2,
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
        wt1=0.8, wt2=0.8,
        svd_components1=25, svd_components2=20
    )

    # plot top canonical correlations in a random batch
    fusor.plot_canonical_correlations(
        svd_components1=50,
        svd_components2=None,
        cca_components=45
    )

    fusor.refine_pivots(
        wt1=0.8, wt2=0.8,
        svd_components1=40, svd_components2=None,
        cca_components=25,
        n_iters=1,
        randomized_svd=False,
        svd_runs=1,
        verbose=True
    )

    fusor.filter_bad_matches(target='pivot', filter_prop=0.35)

    pivot_matching = fusor.get_matching(order=(2, 1), target='pivot')

    if visualize:
        #labels_plot=np.unique(labels_rna[pivot_matching[0]]) if len(np.unique(labels_rna[pivot_matching[0]]))>len(np.unique(labels_codex[pivot_matching[1]])) else np.unique(labels_codex[pivot_matching[1]])
        labels_plot = np.unique(np.concatenate((labels_rna[pivot_matching[0]],labels_codex[pivot_matching[1]])))
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
        wt1=0.8,
        wt2=0.8,
    )

    fusor.filter_bad_matches(
        target='propagated',
        filter_prop=0.15
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
        subs = 3000
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


