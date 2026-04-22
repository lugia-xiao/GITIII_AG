import os
import torch
import numpy as np
import anndata as ad
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from gitiii_ag.run_maxfuse import maxfuse_integration,maxfuse_integration_debugged
from gitiii_ag.pathway_analyze_utils import search_interactions_LR,group_strings_by_numbers,perform_lasso_cv_with_mse,get_distance_scaler,plot_with_correlations,plot_boxplot_and_pvalues,sum_values_by_category,load_dataset

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

plt.rcParams['image.cmap'] = 'jet'

import re
def correct_string(s):
    return re.sub(r'[^0-9a-zA-Z_]', '_', s)

class Pathway_analyzer():
    def __init__(self,sc_adata,st_sample,species="human",sc_label=None,st_label=None,
                 filter_noise_proportion=0.02,discard_no_match_threshold=0.7,num_neighbors=50,
                 genes_to_remove_from_LR=None):
        assert species in ["human", "mouse"], "Species must be one of human or mouse"
        self.species=species

        self.filter_noise_proportion=filter_noise_proportion
        self.discard_no_match_threshold=discard_no_match_threshold

        self.st_sample=st_sample
        self.sample=st_sample
        self.genes = torch.load(os.path.join(os.getcwd(), "data", "target_genes.pth"),weights_only=False)

        all_genes = torch.load(os.path.join(os.getcwd(), "data", "genes.pth"),weights_only=False)
        self.all_st_genes = all_genes
        genes = torch.load(os.path.join(os.getcwd(), "data", "target_genes.pth"),weights_only=False)
        target_gene_index = []
        for i in range(len(all_genes)):
            if all_genes[i] in genes:
                target_gene_index.append(i)
        self.target_gene_index = target_gene_index

        st_df_path=os.path.join(os.getcwd(),"data","processed",st_sample+".csv")
        st_df=pd.read_csv(st_df_path)

        '''
        if 'flag' in st_df.columns.tolist():
            print(st_df.shape)
            print("There are", np.sum(st_df["flag"] == 0), "cells not used for analysis due to they are too far away "
                                                           "from their nearest neighbors. This is due to they are moved "
                                                           "to other places during preparing the slide")
            st_df=st_df[st_df['flag']!=0]
            print(st_df.shape)
        '''


        st_type_exp_path=os.path.join(os.getcwd(),"data","processed",st_sample+"_TypeExp.npz")
        type_exp_dict = np.load(st_type_exp_path, allow_pickle=True)
        X = st_df.loc[:, self.all_st_genes].values
        type_exp = np.stack([type_exp_dict[cell_type] for cell_type in st_df["subclass"]], axis=0)
        X = X + type_exp
        self.st_adata=sc.AnnData(X=X)
        self.st_adata.var_names = self.all_st_genes
        self.st_adata.obs["celltype"] = st_df["subclass"].values
        self.st_adata.obs["barcode"] = st_df.index
        self.st_adata.obs["centerx"] = st_df["centerx"].values
        self.st_adata.obs["centery"] = st_df["centery"].values
        if 'flag' in st_df.columns.tolist():
            self.st_adata.obs["flag"] = st_df["flag"].values
        for i in range(1, num_neighbors):
            self.st_adata.obs["barcode_NN" + str(i)] = st_df["index_" + str(i)].values

        self.sc_adata=sc_adata
        self.sc_label=sc_label
        if st_label is not None:
            self.st_label=st_label
        else:
            self.st_label=self.st_adata.obs["celltype"]

        self.match_df=None
        self.adata=None
        self.adata_receiver=None
        self.current_LR_regress=None
        self.LIME={}
        self.current_LIME=None

        self.result_dir=os.path.join(os.getcwd(),"influence_tensor")
        if self.result_dir[-1]!="/":
            self.result_dir+="/"
        self.data_dir=os.path.join(os.getcwd(),"data","processed")
        if self.data_dir[-1]!="/":
            self.data_dir+="/"

        if not os.path.exists(os.path.join(os.getcwd(), "results")):
            os.mkdir(os.path.join(os.getcwd(), "results"))

        self.all_genes = list(set(self.sc_adata.var_names.tolist()).union(set(self.st_adata.var_names.tolist())))
        self.inferred_genes=[i for i in self.all_genes if i not in self.genes]
        self.all_genes = self.genes + self.inferred_genes
        self.all_genes=[i for i in self.all_genes if i not in self.sc_adata.var_names.tolist()]+[i for i in self.all_genes if i in self.sc_adata.var_names.tolist()]

        print("Union of genes in scRNA-seq and image-based ST:",len(self.all_genes))

        cell_types = torch.load(os.path.join(os.getcwd(), "data", "processed", "cell_types.pth"))
        self.cell_types = cell_types

        print("Searching for ligand-receptor gene pairs")

        genes_to_search=self.all_genes
        if genes_to_remove_from_LR is not None:
            genes_to_search=[g for g in self.all_genes if g not in genes_to_remove_from_LR]
        self.interactions=search_interactions_LR(genes=genes_to_search,species=species)

        self.ligands_info = self.interactions[0]
        self.ligands_index = []
        for i in range(len(self.ligands_info[0])):
            ligandi = self.ligands_info[0][i]
            ligand_groupi = self.ligands_info[1][i]
            uniquei, groupi = group_strings_by_numbers(ligandi, ligand_groupi)
            groupi = [np.array([self.all_genes.index(groupi[j][k]) for k in range(len(groupi[j]))]) for j in
                      range(len(groupi))]
            self.ligands_index.append(groupi)

        self.receptors_info = self.interactions[1]
        self.receptors_index = []
        for i in range(len(self.receptors_info[0])):
            receptori = self.receptors_info[0][i]
            receptor_groupi = self.receptors_info[1][i]
            uniquei, groupi = group_strings_by_numbers(receptori, receptor_groupi)
            groupi = [np.array([self.all_genes.index(groupi[j][k]) for k in range(len(groupi[j]))]) for j in
                      range(len(groupi))]
            self.receptors_index.append(groupi)

        #print(self.ligands_index)
        #print(self.receptors_index)

        self.signal_names = []
        for i in range(len(self.receptors_info[0])):
            ligandi = self.ligands_info[0][i]
            receptori = self.receptors_info[0][i]
            self.signal_names.append("-".join(ligandi) + "->" + "-".join(receptori))

        self.pathway_names = self.interactions[2]
        print("Finish searching for LR pairs")

        print("Loading and pre-processing influence tensor")
        # Load the results
        self.results = torch.load(os.path.join(self.result_dir, "edges_" + self.sample + ".pth"),weights_only=False)
        # Extract relevant data
        self.attention_scores = self.results["attention_score"]  # Shape (B, 49, C)
        self.indices_all = self.results["NN"]

        # Filter out noise influence
        proportion = torch.abs(self.attention_scores)
        proportion = proportion / torch.sum(proportion, dim=1, keepdim=True)
        self.attention_scores[proportion < self.filter_noise_proportion] = 0

        # get proportion
        proportion = torch.abs(self.attention_scores)
        self.proportion = proportion / torch.sum(proportion, dim=1, keepdim=True)

        # calculate average proportion for future filter out cells that do not have scRNA-seq match by MaxFuse
        self.avg_proportion = torch.mean(torch.abs(self.attention_scores) / torch.sum(torch.abs(self.attention_scores), dim=1, keepdim=True), dim=-1)

        # convert to absolute expression
        self.results["y_state"] = self.results["y"]
        self.cell_type_names = np.array(self.results["cell_type_name"])
        self.cell_type_target = [self.cell_type_names[i][0] for i in range(len(self.cell_type_names))]
        self.type_exp_dict = np.load(self.data_dir + self.sample + "_TypeExp.npz", allow_pickle=True)

        type_exps = torch.Tensor(np.stack([self.type_exp_dict[cell_typei][np.array(self.target_gene_index)] for cell_typei in self.cell_type_target], axis=0))
        self.results["y"] = self.results["y"] + type_exps

        self.results["y_state_pred"] = self.results["y_pred"]
        self.results["y_pred"]=self.results["y_pred"]+type_exps

        self.position_xs = self.results["position_x"][:, 1:]
        self.position_ys = self.results["position_y"][:, 1:]
        self.position_x0 = self.results["position_x"][:, 0:1]
        self.position_y0 = self.results["position_y"][:, 0:1]
        self.distances = torch.sqrt(torch.square(self.position_xs - self.position_x0) + torch.square(self.position_ys - self.position_y0))
        print("Finish loading and pre-processing influence tensor")

        # Auto-detect if the input imaging-based ST already have full transcriptomics coverage
        if len(torch.load(os.path.join(os.getcwd(), "data", "genes.pth"),weights_only=False))>10000:
            print("Good, you already have over 10000 genes in your single-cell-level spatial transcriptomics dataset,\n no need to do full transcriptomics gene imputation")
            #print(set(self.all_genes).difference(set(self.st_adata.var_names.tolist())))
            self.adata=self.st_adata[:,self.all_genes].copy()
            self.adata.obs["have_match"]=[True for i in range(self.adata.shape[0])]


    def maxfuse_integrate(self,common_genes=None,to_csv_path=None,visualize=False,label_path=None,
                          library_size_normalize_sc=False,log_normalize_sc=False,scale_normalize_sc=False,
                          use_my_debugged_function_of_wrong_maxfuse_code=True,
                          metacell_size=2,
                          init_wt1=0.3, init_wt2=0.3,
                          refine_wt1=0.3, refine_wt2=0.3,
                          propagate_wt1=0.7, propagate_wt2=0.7,
                          refine_filter_prop=0.1, check=False
                          ):
        """
        Integrate RNA and protein data using the MaxFuse method.

        Parameters:
        common_genes : list, optional
            List of common genes to use for the integration. If None, it will automatically determine the common genes.
        to_csv_path : str, optional
            File path to save the integrated data as a CSV file. If None, the data will not be saved.
        visualize : bool, optional
            If True, generates a visualization of the integration process.
        label_path: str, optional
            The path to save the cell type labels for multi-modal data integration.

        Returns:
        Dataframe: the alignment of cells between two modalities
        """
        if common_genes is None:
            common_genes=list(set(self.sc_adata.var_names.tolist()).intersection(set(self.st_adata.var_names.tolist())))

        if to_csv_path is None:
            to_csv_path = os.path.join(os.getcwd(), "data", "match.csv")
        if use_my_debugged_function_of_wrong_maxfuse_code:
            self.match_df = maxfuse_integration_debugged(common_genes=common_genes, rna_adata=self.sc_adata.copy(),
                                                         protein_adata=self.st_adata.copy(), to_csv_path=to_csv_path,
                                                         labels_rna=self.sc_label, labels_codex=self.st_label,
                                                         visualize=visualize, labels_path=label_path,
                                                         library_size_normalize_sc=library_size_normalize_sc,
                                                         log_normalize_sc=log_normalize_sc,
                                                         scale_normalize_sc=scale_normalize_sc,
                                                         metacell_size=metacell_size,init_wt1=init_wt1,init_wt2=init_wt2,
                                                         refine_wt1=refine_wt1,refine_wt2=refine_wt2,
                                                         propagate_wt1=propagate_wt1,propagate_wt2=propagate_wt2,
                                                         refine_filter_prop=refine_filter_prop)
        else:
            self.match_df = maxfuse_integration(common_genes=common_genes, rna_adata=self.sc_adata.copy(),
                                                protein_adata=self.st_adata.copy(), to_csv_path=to_csv_path,
                                                labels_rna=self.sc_label, labels_codex=self.st_label,
                                                visualize=visualize, labels_path=label_path,
                                                library_size_normalize_sc=library_size_normalize_sc,
                                                log_normalize_sc=log_normalize_sc,
                                                scale_normalize_sc=scale_normalize_sc,
                                                metacell_size=metacell_size,init_wt1=init_wt1,init_wt2=init_wt2,
                                                refine_wt1=refine_wt1,refine_wt2=refine_wt2,
                                                propagate_wt1=propagate_wt1,propagate_wt2=propagate_wt2,
                                                refine_filter_prop=refine_filter_prop)
        self.integrate_sc_st_with_match_df(self.match_df, check=check)

    def integrate_sc_st_with_match_df(self,match_df, check=False):
        self.match_df=match_df
        self.sc_adata = self.sc_adata[:, [i for i in self.all_genes if i in self.sc_adata.var_names.tolist()]].copy()
        if not isinstance(self.sc_adata.X, np.ndarray):
            self.sc_adata.X = self.sc_adata.X.toarray()

        data_matrix = np.zeros((self.st_adata.shape[0], self.sc_adata.X.shape[1]))
        for i in list(self.match_df.index):
            data_matrix[int(self.match_df.loc[i, "mod2_indx"]), :] = self.sc_adata.X[int(self.match_df.loc[i, "mod1_indx"]), :]

        matched_st = np.unique(self.match_df["mod2_indx"].astype(int))
        have_match_flag = [i in matched_st for i in range(self.st_adata.shape[0])]#[i in self.match_df.index for i in range(self.st_adata.shape[0])]

        all_data_matrix = np.concatenate([self.st_adata[:,[i for i in self.all_genes if i not in self.sc_adata.var_names.tolist()]].X, data_matrix], axis=-1)

        all_data_matrix=np.concatenate([all_data_matrix,self.st_adata.X], axis=-1)

        self.adata = ad.AnnData(X=all_data_matrix, obs=self.st_adata.obs.copy())
        self.adata.obs["have_match"] = have_match_flag
        print("Have match", np.sum(have_match_flag),"/",self.adata.shape[0])
        self.adata.var_names = [i for i in self.all_genes if i not in self.sc_adata.var_names.tolist()]+[i for i in self.all_genes if i in self.sc_adata.var_names.tolist()]+[i+"_st" for i in self.st_adata.var_names.tolist()]

        if not check:
            return
        print("Doing imputation checking")
        cell_type_key='celltype' if 'celltype' in self.adata.obs.columns.tolist() else 'subclass'
        matched_st = list(np.unique(self.match_df["mod2_indx"].astype(int)))
        adata_imputed=self.adata[matched_st, [i for i in self.all_genes if i in self.sc_adata.var_names.tolist()]].copy()
        sc.pp.highly_variable_genes(adata_imputed, n_top_genes=2000, subset=True)
        sc.pp.pca(adata_imputed)
        sc.pp.neighbors(adata_imputed,n_neighbors=40)
        sc.tl.umap(adata_imputed)
        sc.pl.umap(
            adata_imputed,
            color=cell_type_key,
            title="Imputed gene by cell type",
            show=True
        )

        adata_original = self.adata[:,[i+"_st" for i in self.st_adata.var_names.tolist()]].copy()
        sc.pp.pca(adata_original)
        sc.pp.neighbors(adata_original)
        sc.tl.umap(adata_original)
        sc.pl.umap(
            adata_original,
            color=cell_type_key,
            title="Original ST gene by cell type",
            show=True
        )

        intersection_genes=list(set(self.st_adata.var_names).intersection(set(self.sc_adata.var_names)))
        data_sc=self.adata[:,intersection_genes].X.copy()
        data_st=self.adata[:,[str(i)+"_st" for i in intersection_genes]].X.copy()

        assert data_sc.shape == data_st.shape
        n_cells, n_genes = data_sc.shape

        pcc_list = np.empty(n_genes)
        scc_list = np.empty(n_genes)

        for j in range(n_genes):
            x = data_sc[:, j]
            y = data_st[:, j]
            pcc_list[j], _ = pearsonr(x, y)
            scc_list[j], _ = spearmanr(x, y)

        # Plot PCC distribution
        plt.figure(figsize=(7, 5))
        plt.hist(pcc_list, bins=20, color='skyblue', edgecolor='black')
        plt.xlabel('Pearson correlation coefficient')
        plt.ylabel('Number of genes')
        plt.title('Distribution of per-gene PCC between data_sc and data_st')
        plt.tight_layout()
        plt.show()

        # Plot SCC distribution
        plt.figure(figsize=(7, 5))
        plt.hist(scc_list, bins=20, color='salmon', edgecolor='black')
        plt.xlabel('Spearman correlation coefficient')
        plt.ylabel('Number of genes')
        plt.title('Distribution of per-gene SCC between data_sc and data_st')
        plt.tight_layout()
        plt.show()



    def calculate_LR_for_one_cell_using_distance_scaler(self, indices, distance_scalers,
                                                  filter_threshold=0.7):
        '''

        :param indices: length k+1 (default 50)
        :param distance_scalers: length N, N is number of neighbor
        :param filter_threshold if the sum of proportional distance scaler of all this receiver cell's
        sender cells that have scRNA-seq match do not exceed the threshold, then this receiver cell is
        discarded. (flag is False, else flag is True)
        :return:
        '''
        assert len(indices) - 1 == len(distance_scalers)

        receiver_cell = self.adata.X[indices[0], :]
        sender_cells = self.adata.X[indices[1:], :]

        # calculate ligand scores
        data_ligand = []
        for i in range(len(self.ligands_info[0])):
            data_ligandi = np.ones((sender_cells.shape[0]))
            for j in range(len(self.ligands_index[i])):
                data_ligandi = data_ligandi * np.mean(sender_cells[:,self.ligands_index[i][j]], axis=-1)
            data_ligandi = np.power(data_ligandi, 1 / len(self.ligands_index[i]))
            data_ligand.append(data_ligandi)
        data_ligand = np.array(data_ligand)* np.expand_dims(distance_scalers, axis=0)
        data_ligand = np.sum(data_ligand, axis=-1)

        # calculate receptor scores
        data_receptor = []
        for i in range(len(self.receptors_info[0])):
            data_receptori = 1
            for j in range(len(self.receptors_index[i])):
                data_receptori = data_receptori * np.mean(receiver_cell[self.receptors_index[i][j]], axis=-1)
            data_receptori = np.power(data_receptori, 1 / len(self.receptors_index[i]))
            data_receptor.append(data_receptori)
        data_receptor = np.array(data_receptor)

        # calculate filtering threshold
        weights_sum = np.sum(distance_scalers)
        weights = np.sum(self.adata.obs["have_match"][indices[1:]] * distance_scalers)
        flag = weights / weights_sum > filter_threshold and self.adata.obs["have_match"][indices[0]]
        return data_ligand, data_receptor, flag

    def get_distance_scaler(self,receiver_type,targeted_gene="all",count_threshold=100,visualize=True):
        return get_distance_scaler(genei=targeted_gene, genes=self.genes, results=self.results,
                                   attention_scores=self.attention_scores, distances=self.distances,
                                   receiver_type=receiver_type, count_threshold=count_threshold,
                                   visualize=visualize)


    def find_significant_LR_with_known_receiver_value__scaled_LR_VS_cell_state(self,
                                                                               receiver_type, targeted_gene,
                                                                               visualize=True,max_iter=5000,
                                                                               return_LR=False):
        self.current_LR_regress=receiver_type+"--"+targeted_gene
        print("Now calculating which pathway significantly influence the gene of", targeted_gene, "in ", receiver_type)
        print("scaled_LR_VS_cell_state")

        # Calculate the ligand-receptor signaling strength for each cell with respect to the target gene of:
        idx = self.genes.index(targeted_gene)
        proportioni = self.get_distance_scaler(receiver_type=receiver_type,targeted_gene=targeted_gene,visualize=visualize)

        data_ligandi = []
        data_receptori = []
        flagi = []

        for j in range(proportioni.shape[0]):
            if self.cell_type_target[j] == receiver_type:
                data_ligandj, data_receptorj, flagj = self.calculate_LR_for_one_cell_using_distance_scaler(
                    indices=self.indices_all[j], distance_scalers=proportioni[j],
                    filter_threshold=self.discard_no_match_threshold)
                data_ligandi.append(data_ligandj)
                data_receptori.append(data_receptorj)
                flagi.append(flagj)
            if j % 1000 == 0:
                print(j, self.st_adata.shape[0])

        data_ligandi = np.stack(data_ligandi, axis=0)
        data_receptori = np.stack(data_receptori, axis=0)
        flagi = np.array(flagi)
        signal_strengthi = data_ligandi * data_receptori
        print("Number of LR found in scRNA-seq:",signal_strengthi.shape[1])

        type_flag = np.array([j == receiver_type for j in self.cell_type_target])
        y = self.results["y"].numpy()[type_flag, idx]
        y_pred_DL=self.results["y_pred"].numpy()[type_flag, idx]
        # done

        flag_ij = np.array(
            flagi)  # np.logical_and(np.array(flagi), np.array([j == receiver_type for j in cell_type_target]))

        signal_strengthi = signal_strengthi[flag_ij]
        y = y[flag_ij]
        y_pred_DL=y_pred_DL[flag_ij]
        print("Doing LASSO analysis")

        if return_LR:
            return signal_strengthi,y,y_pred_DL

        weights=np.abs(y-y_pred_DL)
        weights=weights/np.max(weights)
        weights=np.exp(-weights)

        indices_to_zero_order, ranked_indices, coefficients, best_alpha, y_pred = perform_lasso_cv_with_mse(X=signal_strengthi, y=y,
                                                                                     feature_names=self.signal_names,weights=weights,
                                                                                     max_iter=max_iter,visualize=visualize)
        torch.save([self.signal_names, indices_to_zero_order, ranked_indices, coefficients, best_alpha, y_pred],
                   os.path.join(os.getcwd(), "results", correct_string(str(receiver_type)+"__" + str(targeted_gene) + "_method1.pth")))

        significant_LR = np.array(self.signal_names)[ranked_indices]
        significant_pathway = np.array(self.pathway_names)[ranked_indices]
        print("These pathway significantly influence the gene of", targeted_gene, "in ", receiver_type, ":")
        print(len(significant_LR.tolist()), significant_LR.tolist())
        print("They are in the following pathways:", np.unique(significant_pathway, return_counts=True))

        reciver_indices = self.indices_all[np.array(np.array(self.cell_type_target) == receiver_type), 0][flag_ij]
        print(reciver_indices.shape)
        self.adata_receiver = self.adata[reciver_indices.flatten(), :].copy()
        for i in range(len(significant_LR.tolist())):
            self.adata_receiver.obs[targeted_gene + "--" + significant_LR.tolist()[i]] = signal_strengthi[:,
                                                                                         self.signal_names.index(
                                                                                             significant_LR.tolist()[
                                                                                                 i])]

        if visualize and len(significant_LR.tolist())>0:
            plot_with_correlations(x=y_pred, y=y, label_x="LASSO predicted expression", label_y="Measured expression")
            plot_with_correlations(x=y_pred_DL, y=y, label_x="DL predicted expression", label_y="Measured expression")
            plot_with_correlations(x=y_pred, y=y_pred_DL, label_x="LASSO predicted expression",
                                   label_y="DL predicted expression")

        if visualize and len(significant_LR.tolist())>0:
            self.adata_receiver.obs[targeted_gene + "_pred_lasso"] = y_pred
            self.adata_receiver.obs[targeted_gene + "_pred_DL"] = y_pred_DL

            if targeted_gene in self.adata_receiver.var_names.tolist() and targeted_gene + "_st" in self.adata_receiver.var_names.tolist():
                sc.pl.scatter(
                    self.adata_receiver,
                    x="centerx",  # 'position_x',
                    y="centery",  # 'position_y',
                    color=targeted_gene,
                    title=f"Spatial expression of {targeted_gene} by scRNA-seq"
                )
                sc.pl.scatter(
                    self.adata_receiver,
                    x="centerx",  # 'position_x',
                    y="centery",  # 'position_y',
                    color=targeted_gene + "_st",
                    title=f"Spatial expression of {targeted_gene} by image-based ST"
                )
            else:
                sc.pl.scatter(
                    self.adata_receiver,
                    x="centerx",  # 'position_x',
                    y="centery",  # 'position_y',
                    color=targeted_gene,
                    title=f"Spatial expression of {targeted_gene}"
                )

            sc.pl.scatter(
                self.adata_receiver,
                x="centerx",  # 'position_x',
                y="centery",  # 'position_y',
                color=targeted_gene + "_pred_DL",
                title=f"DL predicted spatial expression of {targeted_gene}"
            )
            sc.pl.scatter(
                self.adata_receiver,
                x="centerx",  # 'position_x',
                y="centery",  # 'position_y',
                color=targeted_gene + "_pred_lasso",
                title=f"Lasso predicted spatial expression of {targeted_gene}"
            )

        return significant_LR.tolist(), np.array(self.signal_names)[indices_to_zero_order].tolist()

    def find_significant_LR_with_known_receiver_value__scaled_LR_VS_cell_state_multi_sample_first_half(self,
                                                                               receiver_type, targeted_gene):
        self.current_LR_regress=receiver_type+"--"+targeted_gene
        print("Now calculating which pathway significantly influence the gene of", targeted_gene, "in ", receiver_type)
        print("scaled_LR_VS_cell_state")

        # Calculate the ligand-receptor signaling strength for each cell with respect to the target gene of:
        idx = self.genes.index(targeted_gene)
        proportioni = self.get_distance_scaler(receiver_type=receiver_type,targeted_gene=targeted_gene)

        data_ligandi = []
        data_receptori = []
        flagi = []

        for j in range(proportioni.shape[0]):
            if self.cell_type_target[j] == receiver_type:
                data_ligandj, data_receptorj, flagj = self.calculate_LR_for_one_cell_using_distance_scaler(
                    indices=self.indices_all[j], distance_scalers=proportioni[j],
                    filter_threshold=self.discard_no_match_threshold)
                data_ligandi.append(data_ligandj)
                data_receptori.append(data_receptorj)
                flagi.append(flagj)
            if j % 1000 == 0:
                print(j, self.st_adata.shape[0])

        data_ligandi = np.stack(data_ligandi, axis=0)
        data_receptori = np.stack(data_receptori, axis=0)
        flagi = np.array(flagi)
        signal_strengthi = data_ligandi * data_receptori
        print("Number of LR found in scRNA-seq:",signal_strengthi.shape[1])

        type_flag = np.array([j == receiver_type for j in self.cell_type_target])
        y = self.results["y"].numpy()[type_flag, idx]
        y_pred_DL=self.results["y_pred"].numpy()[type_flag, idx]
        # done

        flag_ij = np.array(
            flagi)  # np.logical_and(np.array(flagi), np.array([j == receiver_type for j in cell_type_target]))

        signal_strengthi = signal_strengthi[flag_ij]
        y = y[flag_ij]
        y_pred_DL=y_pred_DL[flag_ij]
        return signal_strengthi,y,y_pred_DL

    def visualize_calculate_LR(self,ligand_genes,receptor_genes,receiver_cell_type,target_gene,normalize=None,return_y=False):
        # Search for this LR given the list of ligand genes and receptor genes
        if self.species=='human':
            database = load_dataset('interactions_human_nonichenetv2')
        else:
            database = load_dataset('interactions_mouse_nonichenetv2')

        LR=None
        for LRi in database:
            ligandi=LRi[0]
            receptori=LRi[2]

            flag_ligand= len(ligand_genes)==len(ligandi) and np.sum([i in ligandi for i in ligand_genes])==len(ligand_genes)
            flag_receptor= len(receptor_genes)==len(receptori) and np.sum([i in receptori for i in receptor_genes])==len(receptor_genes)
            if flag_ligand and flag_receptor:
                LR=LRi
                break

        if LR is None:
            raise ValueError("Input LR does not exist in the database")
        print("Find LR:",LR)

        proportioni = self.get_distance_scaler(receiver_type=receiver_cell_type, targeted_gene=target_gene)

        flags=[]
        LR_datas=[]

        for cell_index in range(proportioni.shape[0]):
            if self.cell_type_target[cell_index] == receiver_cell_type:
                indices = self.indices_all[cell_index]
                distance_scalers = proportioni[cell_index]
                filter_threshold = self.discard_no_match_threshold

                assert len(indices) - 1 == len(distance_scalers)

                receiver_cell = self.adata[indices[0]:indices[0]+1, :]
                sender_cells = self.adata[indices[1:], :]

                data_ligandi = np.ones((sender_cells.shape[0]))
                for ligand_groupi in np.unique(LR[1]):
                    ligand_genes_in_this_group=np.array(LR[0])[np.array([i for i in range(len(LR[0])) if LR[1][i]==ligand_groupi])]
                    data_ligandi=data_ligandi*np.mean(sender_cells.X[:,np.array([sender_cells.var_names.tolist().index(i) for i in ligand_genes_in_this_group])],axis=-1)

                data_ligandi=np.power(data_ligandi,1/len(np.unique(LR[1])))
                data_ligandi=np.sum(data_ligandi*distance_scalers,axis=0)

                data_receptori=1
                for receptor_groupi in np.unique(LR[3]):
                    receptor_gene_in_this_group=np.array(LR[2])[np.array([i for i in range(len(LR[2])) if LR[3][i]==receptor_groupi])]
                    data_receptori=data_receptori*np.mean(receiver_cell.X[:,np.array([receiver_cell.var_names.tolist().index(i) for i in receptor_gene_in_this_group])],axis=-1)
                data_receptori=np.power(data_receptori,1/len(np.unique(LR[3])))
                data_LRi=data_ligandi*data_receptori

                weights_sum = np.sum(distance_scalers)
                weights = np.sum(self.adata.obs["have_match"][indices[1:]] * distance_scalers)
                flag = weights / weights_sum > filter_threshold and self.adata.obs["have_match"][indices[0]]

                LR_datas.append(data_LRi)
                flags.append(flag)

        LR_datas = np.array(LR_datas)
        flags = np.array(flags)

        reciver_indices = self.indices_all[np.array(np.array(self.cell_type_target) == receiver_cell_type), 0][flags]
        self.adata_receiver = self.adata[reciver_indices.flatten(), :].copy()

        type_flag = np.array([j == receiver_cell_type for j in self.cell_type_target])
        idx = self.genes.index(target_gene)

        y_pred_DL = self.results["y_pred"].numpy()[type_flag, idx]
        y_pred_DL = y_pred_DL[flags]

        y = self.results["y"].numpy()[type_flag, idx]
        y=y[flags]

        strengths=LR_datas[flags]
        if normalize is not None:
            strengths=normalize(strengths)

        self.adata_receiver.obs['-'.join(ligand_genes)+'->'+'-'.join(receptor_genes)+'-->'+target_gene] = strengths
        self.adata_receiver.obs["DL-predicted expression of "+target_gene] = y_pred_DL

        self.adata_receiver.obs['centerx'] = self.adata_receiver.obs['centerx'] - np.min(
            self.adata_receiver.obs['centerx'])
        self.adata_receiver.obs['centery'] = self.adata_receiver.obs['centery'] - np.min(
            self.adata_receiver.obs['centery'])

        sc.pl.scatter(
            self.adata_receiver,
            x="centerx",  # 'position_x',
            y="centery",  # 'position_y',
            color='-'.join(ligand_genes)+'->'+'-'.join(receptor_genes)+'-->'+target_gene,
            title='-'.join(ligand_genes)+'->'+'-'.join(receptor_genes)+'-->'+target_gene
        )

        if return_y:
            sc.pl.scatter(
                self.adata_receiver,
                x="centerx",  # 'position_x',
                y="centery",  # 'position_y',
                color=target_gene,
                title="Measured expression of "+target_gene
            )

            sc.pl.scatter(
                self.adata_receiver,
                x="centerx",  # 'position_x',
                y="centery",  # 'position_y',
                color="DL-predicted expression of "+target_gene,
                title="DL-predicted expression of "+target_gene
            )

            try:
                print(strengths.shape,y_pred_DL.shape,y.shape)
                plot_with_correlations(
                    x=strengths.flatten(),
                    y=y_pred_DL.flatten(),
                    label_x='-'.join(ligand_genes) + '->' + '-'.join(receptor_genes) + '-->' + target_gene,
                    label_y="DL-predicted expression of " + target_gene
                )
                plot_with_correlations(
                    x=strengths.flatten(),
                    y=y.flatten(),
                    label_x='-'.join(ligand_genes) + '->' + '-'.join(receptor_genes),
                    label_y="Measured expression of " + target_gene
                )
            except Exception as e:
                print("First plot failed:", e)


            yi=np.array(self.adata_receiver[:,target_gene].X).flatten()
            return strengths, yi

        return strengths

    def visualize_LR(self,ligand_gene,receptor_gene,receiver_cell_type,target_gene):
        assert self.adata is not None, "Need to calculate significant LR for specified receiver cell type and target gene first!"

        # Set the default colormap globally for all plots
        plt.rcParams['image.cmap'] = 'jet'

        if ligand_gene+"->"+receptor_gene in self.signal_names:
            if target_gene+"_st" in self.adata_receiver.obs.columns.tolist():
                y_name=target_gene+"_st"
            else:
                y_name=target_gene
            plot_with_correlations(
                x=np.array(self.adata_receiver.obs[target_gene + "--" + ligand_gene + "->" + receptor_gene]),
                y=np.array(self.adata_receiver[:, y_name].X.flatten()),
                label_x="Signaling strength of " + ligand_gene + "->" + receptor_gene,
                label_y="Measured expression of " + target_gene)
        else:
            print("Wrong pathway input")
            if target_gene + "--" + ligand_gene + "->" + receptor_gene in self.adata_receiver.obs:
                sc.pl.scatter(
                    self.adata_receiver,
                    x="centerx",  # 'position_x',
                    y="centery",  # 'position_y',
                    color=target_gene + "--" + ligand_gene + "->" + receptor_gene
                )
            return

        if len(ligand_gene.split("-"))>=2 or len(receptor_gene.split("-"))>=2:#ligand_gene+"->"+receptor_gene not in self.signal_names:
            print("This function currently only supports the visualization of ligand expression and receptor expression of one-ligand one-receptor pathway")
            if target_gene + "--" + ligand_gene + "->" + receptor_gene in self.adata_receiver.obs:
                sc.pl.scatter(
                    self.adata_receiver,
                    x="centerx",  # 'position_x',
                    y="centery",  # 'position_y',
                    color=target_gene + "--" + ligand_gene + "->" + receptor_gene
                )
            return

        print("This is the visualization of",self.current_LR_regress,"using regression between (product of Ligand-receptor expression and distance scaler) and (cell state expression of the receiver cell type)")
        print("If this is not what you want, run the 'find_significant_LR_with_known_receiver_value__scaled_LR_VS_cell_state' function first")

        cell_types_all = self.adata.obs["celltype"].tolist()
        cell_type_plot = [i if i == receiver_cell_type else "Other" for i in cell_types_all]
        self.adata.obs["cell_type_plot"] = cell_type_plot

        # Create a plot with two scatter plots on the same axes
        fig, ax = plt.subplots()
        sc.pl.scatter(
            self.adata[self.adata.obs["cell_type_plot"] != "Other"],
            x="centerx",  # 'position_x',
            y="centery",  # 'position_y',
            color=receptor_gene,
            ax=ax,
            show=False,
            title=""
        )
        custom_palette = ['pink']
        sc.pl.scatter(
            self.adata[self.adata.obs["cell_type_plot"] == "Other"],
            x="centerx",  # 'position_x',
            y="centery",  # 'position_y',
            color="cell_type_plot",
            ax=ax,
            show=False,
            palette=custom_palette,
            title=""
        )
        # Add the first legend
        handles1, labels1 = ax.get_legend_handles_labels()
        print(ax.get_legend_handles_labels())
        legend1 = ax.legend(handles1[:len(handles1) // 2], labels1[:len(labels1) // 2], loc="upper right",
                            bbox_to_anchor=(1.2, 1), ncol=1)

        # Add the second legend below the first one
        legend2 = ax.legend(handles1[len(handles1) // 2:], labels1[len(labels1) // 2:], loc="upper right",
                            bbox_to_anchor=(1.3, 1), ncol=1)

        # Add the first legend back to the axes
        ax.add_artist(legend1)

        # Modify the axis labels to display 'position_x' and 'position_y'
        # ax.set_xlabel('position x')
        # ax.set_ylabel('position y')

        # Adjust layout to ensure there is no clipping
        ax.set_title(receptor_gene)
        ax.set_aspect("equal", adjustable="datalim")
        plt.tight_layout()
        plt.show()

        sc.pl.scatter(
            self.adata,
            x="centerx",  # 'position_x',
            y="centery",  # 'position_y',
            color=ligand_gene
        )
        if target_gene + "--" + ligand_gene + "->" + receptor_gene in self.adata_receiver.obs:
            sc.pl.scatter(
                self.adata_receiver,
                x="centerx",  # 'position_x',
                y="centery",  # 'position_y',
                color=target_gene + "--" + ligand_gene + "->" + receptor_gene
            )
        else:
            print("This ligand-receptor pair is not a significant pathway")

    def calculate_LR_for_one_neighborhood(self,indices,scalers,influences,specify_sender=False):
        '''

        :param indices: length N+1 (default 50)
        :return:
        '''
        assert len(indices) - 1 == len(scalers)
        assert len(scalers) == influences.shape[0]

        receiver_cell = self.adata.X[indices[0], :]
        sender_cells = self.adata.X[indices[1:], :]

        # calculate ligand scores
        data_ligand = []
        for i in range(len(self.ligands_info[0])):
            data_ligandi = 1
            for j in range(len(self.ligands_index[i])):
                data_ligandi = data_ligandi * np.mean(sender_cells[:, self.ligands_index[i][j]], axis=-1)
            data_ligandi = np.power(data_ligandi, 1 / len(self.ligands_index[i]))
            data_ligand.append(data_ligandi)
        data_ligand = np.stack(data_ligand, axis=-1)

        # calculate receptor scores
        data_receptor = []
        for i in range(len(self.receptors_info[0])):
            data_receptori = 1
            for j in range(len(self.receptors_index[i])):
                data_receptori = data_receptori * np.mean(receiver_cell[self.receptors_index[i][j]], axis=-1)
            data_receptori = np.power(data_receptori, 1 / len(self.receptors_index[i]))
            data_receptor.append(data_receptori)
        data_receptor = np.array(data_receptor)

        # calculate filtering threshold
        flag = self.adata.obs["have_match"][indices[0]]
        flag = np.array([flag for i in range(len(scalers))])
        flag = np.logical_and(flag, self.adata.obs["have_match"][indices[1:]])
        flag = np.logical_and(flag, influences!=0)
        if specify_sender:
            flag = np.logical_and(flag, scalers != 0)
        datai = np.expand_dims(data_receptor, axis=0) * data_ligand
        datai = np.expand_dims(scalers, axis=-1) * datai
        return datai[flag, :], influences[flag]

    def find_significant_LR__LR_VS_predicted_influence__only_receiver_type(self,receiver_type, targeted_gene,
                                                                           max_iter=5000,visualize=True):
        assert targeted_gene in self.genes, "In this method, target gene must be in genes measured in ST"
        self.current_LIME=receiver_type + "--" + targeted_gene

        print("Now calculating which pathway significantly influence the gene of", targeted_gene, "in ", receiver_type)
        print("LR_VS_predicted_influence__only_receiver_type")

        # Calculate the ligand-receptor signaling strength for each cell with respect to the target gene of:
        distance_scaleri = self.get_distance_scaler(receiver_type=receiver_type,targeted_gene=targeted_gene,visualize=visualize)
        influencesi = self.attention_scores.numpy()[:, :, self.genes.index(targeted_gene)]

        signal_strengthi = []
        predicted_influencei = []

        weights=[]
        idx = self.genes.index(targeted_gene)

        for j in range(distance_scaleri.shape[0]):
            if self.cell_type_target[j] == receiver_type:
                dataj = self.calculate_LR_for_one_neighborhood(self.indices_all[j],
                                                               distance_scaleri[j], influencesi[j],
                                                               specify_sender=False)
                signal_strengthi.append(dataj[0])
                predicted_influencei.append(dataj[1])

                weights.append(np.abs(self.results["y"].numpy()[j, idx]-self.results["y_pred"].numpy()[j, idx])*np.ones_like(dataj[1]))
            if j % 1000 == 0:
                print(j, self.st_adata.shape[0])

        signal_strengthi = np.concatenate(signal_strengthi, axis=0)
        print("Found", signal_strengthi.shape[1], "LR in databases")
        predicted_influencei = np.concatenate(predicted_influencei, axis=0)

        weights = np.concatenate(weights, axis=0)
        weights = weights / np.max(weights)
        weights = np.exp(-weights)#*np.abs(predicted_influencei)
        weights=np.ones_like(weights)
        # done

        signal_strengthi = signal_strengthi
        y = predicted_influencei
        print("Doing LASSO analysis")
        indices_to_zero_order, ranked_indices, coefficients, best_alpha, y_pred = perform_lasso_cv_with_mse(signal_strengthi, y, feature_names=self.signal_names,max_iter=max_iter,visualize=visualize, weights=weights)
        torch.save([self.signal_names, indices_to_zero_order, ranked_indices, coefficients, best_alpha, y_pred],os.path.join(os.getcwd(), "results", correct_string(receiver_type+"__"+str(targeted_gene)+"_method2.pth")))

        significant_LR = np.array(self.signal_names)[ranked_indices]
        z_scores = coefficients
        significant_pathway = np.array(self.pathway_names)[ranked_indices]
        print("These pathway significantly influence the gene of", targeted_gene, "in ", receiver_type, ":")
        print(len(significant_LR.tolist()), significant_LR.tolist())
        print("They are in the following pathways:", np.unique(significant_pathway, return_counts=True))

        self.LIME[targeted_gene] = y
        self.LIME[targeted_gene + "_" + "pred"] = y_pred
        for i in range(len(significant_LR.tolist())):
            self.LIME[targeted_gene+"_"+str(significant_LR.tolist()[i])]=signal_strengthi[:,self.signal_names.index(significant_LR.tolist()[i])]

        if visualize:
            plot_with_correlations(x=y_pred,y=y,label_x="Reconstructed influence",label_y="DL-estimated influence")

        return significant_LR.tolist(), np.array(self.signal_names)[indices_to_zero_order].tolist()


    def find_significant_LR__LR_VS_predicted_influence__only_receiver_type_multi_samples_first_half(self,receiver_type, targeted_gene):
        assert targeted_gene in self.genes, "In this method, target gene must be in genes measured in ST"
        self.current_LIME=receiver_type + "--" + targeted_gene
        # Calculate the ligand-receptor signaling strength for each cell with respect to the target gene of:
        distance_scaleri = self.get_distance_scaler(receiver_type=receiver_type,targeted_gene=targeted_gene)
        influencesi = self.attention_scores.numpy()[:, :, self.genes.index(targeted_gene)]

        signal_strengthi = []
        predicted_influencei = []

        weights = []
        idx = self.genes.index(targeted_gene)

        for j in range(distance_scaleri.shape[0]):
            if self.cell_type_target[j] == receiver_type:
                dataj = self.calculate_LR_for_one_neighborhood(self.indices_all[j],
                                                               distance_scaleri[j], influencesi[j],
                                                               specify_sender=False)
                signal_strengthi.append(dataj[0])
                predicted_influencei.append(dataj[1])

                weights.append(
                    np.abs(self.results["y"].numpy()[j, idx] - self.results["y_pred"].numpy()[j, idx]) * np.ones_like(dataj[1]))
            if j % 1000 == 0:
                print(j, self.st_adata.shape[0])

        signal_strengthi = np.concatenate(signal_strengthi, axis=0)
        predicted_influencei = np.concatenate(predicted_influencei, axis=0)

        # done
        signal_strengthi = signal_strengthi
        y = predicted_influencei

        weights = np.concatenate(weights, axis=0)
        weights = weights / np.max(weights)
        weights = np.exp(-weights)# * np.abs(predicted_influencei)
        weights = np.ones_like(weights)
        return signal_strengthi, y, weights

    def find_significant_LR__LR_VS_predicted_influence__receiver_sender(self,receiver_type, sender_type_list,
                                                                     targeted_gene,visualize=True,max_iter=5000):
        print("Now calculating which pathway significantly influence the gene of", targeted_gene, "in ", receiver_type)
        print("LR_VS_predicted_influence__receiver_sender")
        self.current_LIME="Sender cell types in ["+", ".join(sender_type_list)+"] ->"+receiver_type + "--" + targeted_gene

        # Calculate the ligand-receptor signaling strength for each cell with respect to the target gene of:
        idx = self.genes.index(targeted_gene)
        proportioni_unfiltered = self.get_distance_scaler(receiver_type=receiver_type,targeted_gene=targeted_gene)
        influencesi = self.attention_scores.numpy()[:, :, self.genes.index(targeted_gene)]

        ## filter out sender cells whose cell types do not match the given cell types
        type_filter = np.isin(np.array(self.results["cell_type_name"])[:, 1:], sender_type_list)
        proportioni = np.where(type_filter, proportioni_unfiltered, np.zeros_like(proportioni_unfiltered))

        signal_strengthi = []
        predicted_influencei = []

        weights = []
        idx = self.genes.index(targeted_gene)

        for j in range(proportioni.shape[0]):
            if self.cell_type_target[j] == receiver_type:
                dataj = self.calculate_LR_for_one_neighborhood(self.indices_all[j],
                                                               proportioni[j], influencesi[j],
                                                               specify_sender=True)
                signal_strengthi.append(dataj[0])
                predicted_influencei.append(dataj[1])

                weights.append(np.abs(self.results["y"].numpy()[j, idx] - self.results["y_pred"].numpy()[j, idx]) * np.ones_like(dataj[1]))
            if j % 1000 == 0:
                print(j, self.st_adata.shape[0])

        signal_strengthi = np.concatenate(signal_strengthi, axis=0)
        predicted_influencei = np.concatenate(predicted_influencei, axis=0)

        weights = np.concatenate(weights, axis=0)
        weights = weights / np.max(weights)
        weights = np.exp(-weights)# * np.abs(predicted_influencei)
        weights = np.ones_like(weights)
        # done

        signal_strengthi = signal_strengthi
        y = predicted_influencei
        #weights = np.concatenate(weights, axis=0)
        indices_to_zero_order, ranked_indices, coefficients, best_alpha, y_pred = perform_lasso_cv_with_mse(signal_strengthi, y,feature_names=self.signal_names,max_iter=max_iter, weights=weights)

        torch.save([self.signal_names, indices_to_zero_order, ranked_indices, coefficients, best_alpha, y_pred],
                   os.path.join(os.getcwd(), "results", correct_string(str(receiver_type)+"__"+ '-'.join(sender_type_list)+"__" + str(targeted_gene) + "_method2_sender.pth")))

        significant_LR = np.array(self.signal_names)[ranked_indices]
        z_scores = coefficients
        significant_pathway = np.array(self.pathway_names)[ranked_indices]
        print("These pathway significantly influence the gene of", targeted_gene, "in ", receiver_type, ":")
        print(len(significant_LR.tolist()), significant_LR.tolist())
        print("They are in the following pathways:", np.unique(significant_pathway, return_counts=True))

        self.LIME[targeted_gene] = y
        self.LIME[targeted_gene + "_" + "pred"] = y_pred
        for i in range(len(significant_LR.tolist())):
            self.LIME[targeted_gene + "_" + str(significant_LR.tolist()[i])] = signal_strengthi[:, self.signal_names.index(
                significant_LR.tolist()[i])]
        if visualize:
            plot_with_correlations(x=y_pred,y=y,label_x="Reconstructed influence",label_y="DL-estimated influence")
        return significant_LR.tolist(), np.array(self.signal_names)[indices_to_zero_order].tolist()

    def visualize_LR_VS_predicted_influence(self,receiver_type,targeted_gene,pathway_name,sender_type=None):
        if targeted_gene not in self.LIME.keys():
            print("Need to calculate significant LR for specified receiver cell type (and sender cell type) and target gene first!")
            return

        assert pathway_name in self.signal_names, "Pathway name must be in the database"

        print("This is the visualization of", self.current_LIME,
              "using regression between (product of Ligand-receptor expression and distance scaler) and (cell state expression of the receiver cell type)")
        print(
            "If this is not what you want, run the 'find_significant_LR__LR_VS_predicted_influence__receiver_sender' function or the 'find_significant_LR__LR_VS_predicted_influence__only_receiver_type' function first")

        plot_with_correlations(x=self.LIME[targeted_gene + "_" + pathway_name], y=self.LIME[targeted_gene + "_" + "pred"], label_x="Product of LR expression scaled by distance", label_y="DL-estimated influence")

    def calculate_LR_proportion_in_one_neighbor(self, indices, cell_types, distance_scalers, LR,
                                                filter_threshold=0.7,normalize_to_1=True):
        # calculate filtering threshold
        weights_sum = np.sum(distance_scalers)
        weights = np.sum(self.adata.obs["have_match"][indices[1:]] * distance_scalers)
        flag = weights / weights_sum > filter_threshold and self.adata.obs["have_match"][indices[0]]
        if not flag:
            return False,np.zeros((len(self.cell_types)))

        receiver_cell = self.adata.X[indices[0], :]
        sender_cells = self.adata.X[indices[1:], :]

        # find the index of the given LR pair
        i=self.signal_names.index(LR)

        # calculate ligand scores
        data_ligandi = 1
        for j in range(len(self.ligands_index[i])):
            data_ligandi = data_ligandi * np.mean(sender_cells[:, self.ligands_index[i][j]], axis=-1)
        data_ligandi = np.power(data_ligandi, 1 / len(self.ligands_index[i]))

        # calculate receptor scores
        data_receptori = 1
        for j in range(len(self.receptors_index[i])):
            data_receptori = data_receptori * np.mean(receiver_cell[self.receptors_index[i][j]], axis=-1)
        data_receptori = np.power(data_receptori, 1 / len(self.receptors_index[i]))

        datai = np.expand_dims(data_receptori, axis=0) * data_ligandi
        datai = distance_scalers * datai
        datai = datai[self.adata.obs["have_match"][indices[1:]]]
        if not np.sum(datai)>0:
            return False, np.zeros((len(self.cell_types)))
        if normalize_to_1:
            datai = datai / np.sum(datai)
        cell_types = cell_types[self.adata.obs["have_match"][indices[1:]]]
        flagi=True if np.sum(datai)>0 else False
        return flagi,sum_values_by_category(values=datai, categories=cell_types, reference_order=self.cell_types)

    def identify_source_sender_type__known_LR_and_receiver_type(self,receiver_type,LR,target_gene=None,
                                                                filter_threshold=0.7, normalize_to_1=True):
        self.cell_type_names=np.array(self.cell_type_names)
        if target_gene is None:
            target_gene="all"

        assert LR in self.signal_names, "Pathway name must be in the database"
        print("Visualizing",LR)

        # Calculate the ligand-receptor signaling strength for each cell with respect to the target gene of:
        distance_scaleri = self.get_distance_scaler(receiver_type=receiver_type, targeted_gene=target_gene)

        proportions = []
        flags=[]
        for j in range(distance_scaleri.shape[0]):
            if self.cell_type_target[j] == receiver_type:
                cell_types_j=self.cell_type_names[j,1:]
                flagj,dataj = self.calculate_LR_proportion_in_one_neighbor(indices=self.indices_all[j],
                                                                     cell_types=cell_types_j,
                                                                     distance_scalers=distance_scaleri[j],
                                                                     LR=LR,filter_threshold=filter_threshold,
                                                                           normalize_to_1=normalize_to_1)
                proportions.append(dataj)
                flags.append(flagj)
            if j % 5000 == 0:
                print(j, self.st_adata.shape[0])

        proportions = np.stack(proportions, axis=0)
        flags=np.array(flags)
        proportions=proportions[flags]
        plot_boxplot_and_pvalues(data=proportions,feature_names=self.cell_types)

        return proportions








