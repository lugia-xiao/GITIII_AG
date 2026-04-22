import os
import torch
import numpy as np
import anndata as ad
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from gitiii_ag.pathway_analyzer import Pathway_analyzer_prepared,Pathway_analyzer
from gitiii_ag.pathway_analyze_utils import search_interactions_LR,group_strings_by_numbers,perform_lasso_cv_with_mse,get_distance_scaler,plot_with_correlations,plot_boxplot_and_pvalues,sum_values_by_category

class Pathway_analyzer_multi_samples():
    def __init__(self,list_of_pathway_analyzer):
        self.list_of_pathway_analyzer = list_of_pathway_analyzer
        self.signal_names=self.list_of_pathway_analyzer[0].signal_names
        self.pathway_names=self.list_of_pathway_analyzer[0].pathway_names
        self.genes= self.list_of_pathway_analyzer[0].genes

    def find_significant_LR_with_known_receiver_value__scaled_LR_VS_cell_state(self,
                                                                               receiver_type, targeted_gene,
                                                                               max_iter=5000):
        signal_strengths, ys, y_pred_DLs=[], [], []
        for pathway_analyzer in self.list_of_pathway_analyzer:
            signal_strengthi, yi, y_pred_DLi=pathway_analyzer.find_significant_LR_with_known_receiver_value__scaled_LR_VS_cell_state_multi_sample_first_half(
                receiver_type=receiver_type,targeted_gene=targeted_gene
            )
            signal_strengths.append(signal_strengthi)
            ys.append(yi)
            y_pred_DLs.append(y_pred_DLi)

        signal_strengths=np.concatenate(signal_strengths,axis=0)
        ys=np.concatenate(ys,axis=0)
        y_pred_DLs=np.concatenate(y_pred_DLs,axis=0)

        print("Doing LASSO analysis")
        indices_to_zero_order, ranked_indices, coefficients, best_alpha, y_preds = perform_lasso_cv_with_mse(
            X=signal_strengths, y=y_pred_DLs,#ys,
            feature_names=self.signal_names,
            max_iter=max_iter)

        significant_LR = np.array(self.signal_names)[ranked_indices]
        significant_pathway = np.array(self.pathway_names)[ranked_indices]
        print("These pathway significantly influence the gene of", targeted_gene, "in ", receiver_type, ":")
        print(len(significant_LR.tolist()), significant_LR.tolist())
        print("They are in the following pathways:", np.unique(significant_pathway, return_counts=True))

        plot_with_correlations(x=y_preds, y=ys, label_x="LASSO predicted expression", label_y="Measured expression")
        plot_with_correlations(x=y_pred_DLs, y=ys, label_x="DL predicted expression", label_y="Measured expression")
        plot_with_correlations(x=y_preds, y=y_pred_DLs, label_x="LASSO predicted expression",
                               label_y="DL predicted expression")

        return significant_LR.tolist(), np.array(self.signal_names)[indices_to_zero_order].tolist()


    def find_significant_LR__LR_VS_predicted_influence__only_receiver_type(self,receiver_type, targeted_gene,
                                                                           max_iter=5000,visualize=True):
        assert targeted_gene in self.genes, "In this method, target gene must be in genes measured in ST"
        self.current_LIME = receiver_type + "--" + targeted_gene

        print("Now calculating which pathway significantly influence the gene of", targeted_gene, "in ", receiver_type)
        print("LR_VS_predicted_influence__only_receiver_type")

        signal_strengths, ys=[], []
        for pathway_analyzer in self.list_of_pathway_analyzer:
            signal_strengthi, yi=pathway_analyzer.find_significant_LR__LR_VS_predicted_influence__only_receiver_type_multi_samples_first_half(
                receiver_type=receiver_type,targeted_gene=targeted_gene
            )
            signal_strengths.append(signal_strengthi)
            ys.append(yi)

        signal_strengths=np.concatenate(signal_strengths,axis=0)
        ys=np.concatenate(ys,axis=0)

        print("Doing LASSO analysis")
        indices_to_zero_order, ranked_indices, coefficients, best_alpha, y_pred = perform_lasso_cv_with_mse(
            signal_strengths, ys, feature_names=self.signal_names, max_iter=max_iter)

        significant_LR = np.array(self.signal_names)[ranked_indices]
        significant_pathway = np.array(self.pathway_names)[ranked_indices]
        print("These pathway significantly influence the gene of", targeted_gene, "in ", receiver_type, ":")
        print(len(significant_LR.tolist()), significant_LR.tolist())
        print("They are in the following pathways:", np.unique(significant_pathway, return_counts=True))

        if visualize:
            plot_with_correlations(x=y_pred, y=ys, label_x="Reconstructed influence", label_y="DL-estimated influence")

            for i in range(10):
                plot_with_correlations(x=signal_strengths[:, self.signal_names.index(significant_LR.tolist()[i])],
                                       y=ys,
                                       label_x="Product of LR expression scaled by distance",
                                       label_y="DL-estimated influence")
        return significant_LR.tolist(), np.array(self.signal_names)[indices_to_zero_order].tolist()


