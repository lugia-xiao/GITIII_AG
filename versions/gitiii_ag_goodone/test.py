'''
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import anndata

import matplotlib.pyplot as plt

# Set the default colormap globally for all plots
plt.rcParams['image.cmap'] = 'jet'

# Simulate adata with obs containing 'celltype', 'centerx', 'centery', and 'receptor_gene'
n_cells = 500
np.random.seed(0)
cell_types = np.random.choice(['TypeA', 'TypeB', 'TypeC'], size=n_cells, p=[0.1, 0.4, 0.5])
centerx = np.random.rand(n_cells) * 100
centery = np.random.rand(n_cells) * 100
receptor_gene_expression = np.random.rand(n_cells) * 2  # fake gene expression data

obs = pd.DataFrame({
    'celltype': cell_types,
    'centerx': centerx,
    'centery': centery,
    'receptor_gene': receptor_gene_expression
})

# Simulate the AnnData object
adata = anndata.AnnData(obs=obs)

# Define the receiver cell type and receptor gene
receiver_cell_type = "TypeA"
receptor_gene = "receptor_gene"

cell_types_all = adata.obs["celltype"].tolist()
cell_type_plot = [i if i == receiver_cell_type else "Other" for i in cell_types_all]
adata.obs["cell_type_plot"] = cell_type_plot

# Create a plot with two scatter plots on the same axes
fig, ax = plt.subplots()
sc.pl.scatter(
    adata[adata.obs["cell_type_plot"] != "Other"],
    x="centerx",  # 'position_x',
    y="centery",  # 'position_y',
    color='receptor_gene',
    ax=ax,
    show=False,
    title=""
)
custom_palette = ['pink']
sc.pl.scatter(
    adata[adata.obs["cell_type_plot"] == "Other"],
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
legend1 = ax.legend(handles1[:len(handles1)//2], labels1[:len(labels1)//2], loc="upper right", bbox_to_anchor=(1.2, 1), ncol=1)

# Add the second legend below the first one
legend2 = ax.legend(handles1[len(handles1)//2:], labels1[len(labels1)//2:], loc="upper right", bbox_to_anchor=(1.3, 1), ncol=1)

# Add the first legend back to the axes
ax.add_artist(legend1)

# Modify the axis labels to display 'position_x' and 'position_y'
#ax.set_xlabel('position x')
#ax.set_ylabel('position y')

# Adjust layout to ensure there is no clipping
plt.tight_layout()
plt.show()
'''

'''import gitiii_ag
import os
import scanpy as sc

sample='H20.33.001.CX28.MTG.02.007.1.02.03'
sc_adata=sc.read_h5ad("/gpfs/gibbs/project/wang_zuoheng/xx244/GITIII-AG/AD_pipeline/raw_data/sc.h5ad")

pathway_analyzer=gitiii_ag.pathway_analyzer.Pathway_analyzer(sc_adata=sc_adata,st_sample=sample,species="human",sc_label=sc_adata.obs["celltype"],st_label=None,filter_noise_proportion=0.02,discard_no_match_threshold=0.7,num_neighbors=50)

pathway_analyzer.maxfuse_integrate(common_genes=None,to_csv_path=os.path.join(os.getcwd(), "data", "match.csv"),visualize=True,label_path=os.path.join(os.getcwd(),"data","labels.csv"))

significant_LR1,_=pathway_analyzer.find_significant_LR_with_known_receiver_value__scaled_LR_VS_cell_state(receiver_type="L2/3 IT",targeted_gene="RORB",visualize=True)
print(significant_LR1)

for i in range(10):
    LR_pairi=significant_LR1[i]
    ligand_gene=LR_pairi[0].split("-")[0]
    receptor_gene=LR_pairi[1].split("-")[0]
    pathway_analyzer.visualize_LR(ligand_gene=ligand_gene,receptor_gene=receptor_gene,receiver_cell_type="L2/3 IT",target_gene="RORB")

significant_LR2a,_=pathway_analyzer.find_significant_LR__LR_VS_predicted_influence__only_receiver_type(receiver_type="L2/3 IT",targeted_gene="RORB")
print(significant_LR2a)

for i in range(10):
    LR_pairi=significant_LR2a[i]
    pathway_analyzer.visualize_LR_VS_predicted_influence(pathway_name=LR_pairi,receiver_type="L2/3 IT",targeted_gene="RORB")

significant_LR2b,_=pathway_analyzer.find_significant_LR__LR_VS_predicted_influence__receiver_sender(receiver_type="L2/3 IT",targeted_gene="RORB",sender_type="Microglia-PVM")
print(significant_LR2b)

for i in range(10):
    LR_pairi=significant_LR2b[i]
    pathway_analyzer.visualize_LR_VS_predicted_influence(pathway_name=LR_pairi,receiver_type="L2/3 IT",targeted_gene="RORB",sender_type="Microglia-PVM")


sample='H20.33.001.CX28.MTG.02.007.1.02.03'

spatial_visualizer=gitiii_ag.spatial_visualizer.Spatial_visualizer(sample=sample)

spatial_visualizer.plot_distance_scaler(rank_or_distance="distance",proportion_or_abs="abs",target_gene=None,bins=300, frac=0.003)

spatial_visualizer.visualize_CCI_function(select_topk=5,num_type_pair=10)

spatial_visualizer.visualize_information_flow(target_gene="RORB",select_topk=5,use_neuron_layer=True,cutoff=0.05)

subtyping_analyzer=gitiii_ag.subtyping_analyzer.Subtyping_anlayzer(sample=sample,normalize_to_1=True,use_abs=True,noise_threshold=2e-2)

COI="L2/3 IT" # Cell Of Interest
subtyping_analyzer.subtyping(COI=COI,resolution=0.1)

subtyping_analyzer.subtyping_DE()

network_analyzer=gitiii_ag.network_analyzer.Network_analyzer(noise_threshold=2e-2)

network_analyzer.determine_network_sample(sample=sample)'''

'''list1 = ['GLS-SLC17A6-SLC17A7-SLC17A8->GRM7', 'TAC1->TACR1', 'GLS-SLC17A6-SLC17A7-SLC17A8->GRIN2B', 'LAMA5->SV2C', 'NODAL->ACVR1B-ACVR2A', 'PTPRC->CD22', 'NRXN2->NLGN1', 'COMP->CD36', 'WNT8A->FZD6-LRP5', 'LRRC4->NTNG2', 'COL4A6->ITGA3-ITGB1', 'GLS-SLC17A6-SLC17A7-SLC17A8->GRM8', 'SEMA6B->PLXNA2', 'WNT10A->FZD6-LRP5', 'COL4A4->ITGAV-ITGB8', 'TAC3->TACR3', 'LAMA5->ITGA3-ITGB1', 'VTN->ITGAV-ITGB8', 'GLS-SLC17A6-SLC17A7-SLC17A8->GRIN3A', 'GJA1->GJA1', 'MIF->ACKR3', 'LAMA4->SV2C', 'GDF1->ACVR1B-ACVR2A', 'GLS-SLC17A6-SLC17A7-SLC17A8->GRIN2A', 'RETN->CAP1', 'SLURP2->CHRNE', 'POMC->OPRD1', 'COL4A2->CD44', 'C3->CR2', 'COL6A3->GP6', 'QRFP->QRFPR', 'WNT3A->FZD3-LRP6', 'SEMA3C->PLXND1', 'CD226->PVR', 'OSM->OSMR-IL6ST', 'COMP->CD47', 'PECAM1->PECAM1', 'HBEGF->EGFR-ERBB2']

list2 = ['GLS-SLC17A6-SLC17A7-SLC17A8->GRM7', 'TAC1->TACR1', 'GLS-SLC17A6-SLC17A7-SLC17A8->GRIN2B', 'LAMA5->SV2C', 'NODAL->ACVR1B-ACVR2A', 'PTPRC->CD22', 'NRXN2->NLGN1', 'COMP->CD36', 'WNT8A->FZD6-LRP5', 'LRRC4->NTNG2', 'COL4A6->ITGA3-ITGB1', 'GLS-SLC17A6-SLC17A7-SLC17A8->GRM8', 'SEMA6B->PLXNA2', 'WNT10A->FZD6-LRP5', 'COL4A4->ITGAV-ITGB8', 'TAC3->TACR3', 'LAMA5->ITGA3-ITGB1', 'VTN->ITGAV-ITGB8', 'GLS-SLC17A6-SLC17A7-SLC17A8->GRIN3A', 'GJA1->GJA1', 'MIF->ACKR3', 'LAMA4->SV2C', 'GDF1->ACVR1B-ACVR2A', 'GLS-SLC17A6-SLC17A7-SLC17A8->GRIN2A', 'RETN->CAP1', 'SLURP2->CHRNE', 'POMC->OPRD1', 'COL4A2->CD44', 'C3->CR2', 'COL6A3->GP6', 'QRFP->QRFPR', 'WNT3A->FZD3-LRP6', 'SEMA3C->PLXND1', 'CD226->PVR', 'OSM->OSMR-IL6ST', 'COMP->CD47', 'PECAM1->PECAM1', 'HBEGF->EGFR-ERBB2']

# Calculating intersection
intersection = set(list1).intersection(set(list2))
intersection_rate = len(intersection) / len(set(list1).union(set(list2)))

intersection_rate'''

'''import numpy as np
from sklearn.linear_model import Lasso, lasso_path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def perform_lasso_cv_with_mse(X, y, feature_names, mark_topk=5, cv=5, max_iter=20000):
    """
    Perform LASSO regression with custom cross-validation to select the best alpha based on MSE.

    Parameters:
    - X: numpy array of shape (n, c), data matrix
    - y: numpy array of shape (n,), response vector
    - feature_names: list of str, names of the features
    - mark_topk: int, number of top coefficients to annotate on the plot (default: 10)
    - cv: int, number of cross-validation folds (default: 5)

    Returns:
    - ranked_indices: numpy array, indices of selected features ranked by absolute value of coefficients
    - coefficients: numpy array, LASSO coefficients of all features (including zeros)
    - best_alpha: float, best alpha value found using MSE
    """

    # Remove nan values
    X = np.nan_to_num(X)
    y = np.nan_to_num(y)

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define alphas to be tested
    alphas = np.logspace(-4, 0, 50)

    kf = KFold(n_splits=cv, shuffle=True)
    mse_means = []

    for alpha in alphas:
        mse_folds = []
        for train_index, val_index in kf.split(X_scaled):
            X_train_fold, X_val_fold = X_scaled[train_index], X_scaled[val_index]
            y_train_fold, y_val_fold = y[train_index], y[val_index]

            lasso = Lasso(alpha=alpha, max_iter=max_iter)
            lasso.fit(X_train_fold, y_train_fold)
            y_val_pred = lasso.predict(X_val_fold)
            mse_folds.append(np.mean((y_val_fold - y_val_pred) ** 2))

        mse_means.append(np.mean(mse_folds))

    best_alpha_index = np.argmin(mse_means)
    best_alpha = alphas[best_alpha_index]
    print("Best alpha based on MSE:", best_alpha)

    # Fit LASSO with the best alpha
    lasso_best = Lasso(alpha=best_alpha, max_iter=max_iter)
    lasso_best.fit(X_scaled, y)
    y_pred_best = lasso_best.predict(X_scaled)

    # Get the coefficients
    coefficients = lasso_best.coef_

    # Identify the non-zero coefficients and rank them by absolute value
    non_zero_indices = np.where(coefficients != 0)[0]
    non_zero_coefficients = coefficients[non_zero_indices]
    ranked_indices = non_zero_indices[np.argsort(-np.abs(non_zero_coefficients))]

    # Extract top k features with the highest absolute coefficients
    top_k_indices = ranked_indices[:mark_topk]
    top_k_coefficients = coefficients[top_k_indices]
    top_k_feature_names = [feature_names[i] for i in top_k_indices]

    # Plot the mean squared error vs alpha
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(alphas, mse_means, 'k', label='Average MSE across folds')
    plt.axvline(best_alpha, linestyle='--', color='r', label='Best alpha')
    plt.scatter(best_alpha, mse_means[best_alpha_index], color='red')  # Highlight best alpha point
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE vs Alpha')
    plt.legend()

    # Compute the coefficient path
    alphas_lasso, coefs_lasso, _ = lasso_path(X_scaled, y, alphas=alphas, max_iter=5000)

    plt.subplot(1, 2, 2)
    plt.plot(alphas_lasso, coefs_lasso.T)
    plt.axvline(best_alpha, linestyle='--', color='r', label='Best alpha')
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Coefficient Value')
    plt.title('Coefficient Paths')

    # Annotate top k predictors at the best alpha value
    for i, (index, coef) in enumerate(zip(top_k_indices, top_k_coefficients)):
        plt.annotate(top_k_feature_names[i],
                     xy=(best_alpha, coef),
                     xytext=(5, 5),
                     textcoords='offset points',
                     arrowprops=dict(facecolor='black', arrowstyle='->'))

    plt.legend()
    plt.tight_layout()
    plt.show()

    # Sort features based on when their coefficients go to zero
    zero_order = np.sum(coefs_lasso != 0, axis=1)
    sorted_indices = np.argsort(zero_order)[::-1]  # Sort in descending order

    indices_to_zero_order=[]
    for idx in sorted_indices:
        if idx in ranked_indices.tolist():
            indices_to_zero_order.append(idx)
    #indices_to_zero_order=np.array(indices_to_zero_order)
    print("Features in order of when their coefficients go to zero:")
    for idx in sorted_indices:
        if zero_order[idx] > 0:
            print(f"{feature_names[idx]} goes to zero at alpha = {alphas_lasso[::-1][zero_order[idx] - 1]:.4f}")
    print(np.array(feature_names)[indices_to_zero_order])
    return indices_to_zero_order, ranked_indices, coefficients, best_alpha, y_pred_best


# Simulate a dataset for debugging
np.random.seed(0)
X_sim = np.random.randn(1000, 50)
y_sim = X_sim[:, 0] + 2 * X_sim[:, 1] + np.random.randn(1000)
feature_names_sim = [f"Feature_{i}" for i in range(X_sim.shape[1])]

# Run the function with simulated data
indices_to_zero_order,ranked_indices, coefficients, best_alpha, y_pred_best = perform_lasso_cv_with_mse(X_sim, y_sim, feature_names_sim)
'''

'''import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mannwhitneyu


def boxplot_violin_and_pvalues(data, feature_names):
    """
    Creates a box plot and violin plot for each feature in the data,
    prints a DataFrame of p-values from Mann-Whitney U tests between each pair of columns,
    and prints the p-values between the highest mean column and other columns.

    Parameters:
    data (numpy.ndarray): The input data matrix with shape (n, t).
    feature_names (numpy.ndarray): Array of feature names with length t.

    Returns:
    pvalues_df (pd.DataFrame): A DataFrame of p-values from pairwise tests.
    """
    # Step 1: Create a box plot and violin plot for each feature
    plt.figure(figsize=(12, 6))

    # Violin plot
    plt.violinplot(data, showmeans=False, showmedians=True)

    # Overlay with box plot
    plt.boxplot(data, vert=True, patch_artist=True, widths=0.15, positions=np.arange(1, len(feature_names) + 1))

    # Set plot labels and title
    plt.xticks(ticks=np.arange(1, len(feature_names) + 1), labels=feature_names, rotation=45)
    plt.xlabel("Feature")
    plt.ylabel("Value")
    plt.title("Box Plot and Violin Plot of Each Feature")
    plt.show()

    # Step 2: Calculate p-values for pairwise Mann-Whitney U tests
    num_features = data.shape[1]
    pvalues_matrix = np.ones((num_features, num_features))  # Initialize p-values matrix with 1s

    for i in range(num_features):
        for j in range(i + 1, num_features):
            _, p_value = mannwhitneyu(data[:, i], data[:, j], alternative='two-sided')
            pvalues_matrix[i, j] = p_value
            pvalues_matrix[j, i] = p_value  # Mirror the value for symmetry

    # Create a DataFrame for the p-values with feature names as row and column labels
    pvalues_df = pd.DataFrame(pvalues_matrix, index=feature_names, columns=feature_names)

    # Print the full pairwise p-values DataFrame
    print("P-Values for Mann-Whitney U Tests Between Each Pair of Features:")
    print(pvalues_df)

    # Step 3: Identify the highest mean column and compute p-values against other columns
    mean_values = np.mean(data, axis=0)
    highest_mean_index = np.argmax(mean_values)
    highest_mean_feature = feature_names[highest_mean_index]

    print(f"\nThe column with the highest mean is: {highest_mean_feature}")

    # Calculate and print p-values between the highest mean column and each other column
    pvalues_highest_mean = {}
    for i in range(num_features):
        if i != highest_mean_index:
            _, p_value = mannwhitneyu(data[:, highest_mean_index], data[:, i], alternative='two-sided')
            pvalues_highest_mean[feature_names[i]] = p_value

    # Display the p-values as a DataFrame
    pvalues_highest_mean_df = pd.DataFrame.from_dict(pvalues_highest_mean, orient='index',
                                                     columns=[f"P-Value with {highest_mean_feature}"])
    print("\nP-Values between the highest mean column and other columns:")
    print(pvalues_highest_mean_df)

    return pvalues_df, pvalues_highest_mean_df

# Example usage:
data = np.random.randn(1000, 5)  # Replace with your actual data
feature_names = np.array(['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5'])
pvalues_df = boxplot_violin_and_pvalues(data, feature_names)'''

'''import numpy as np
import pandas as pd


def sum_values_by_category(values, categories, reference_order):
    """
    Sums up values according to categories and orders the result based on a reference list.

    Parameters:
    - values: np.array of values to sum.
    - categories: np.array of categories corresponding to each value.
    - reference_order: list of categories specifying the desired order of results.

    Returns:
    - np.array of summed values ordered by reference_order.
    """
    # Create a DataFrame for easy grouping and summing
    df = pd.DataFrame({'Category': categories, 'Value': values})

    # Sum values by category
    summed_values = df.groupby('Category')['Value'].sum()

    # Reindex based on the reference order, filling missing categories with 0
    summed_values = summed_values.reindex(reference_order, fill_value=0)

    # Convert to numpy array
    return summed_values.values


# Example usage
values = np.array([1.5, 2.5, 3.0, 1.2, 0.5])
categories = np.array(['A', 'B', 'A', 'C', 'B'])
reference_order = ['A', 'B', 'C', 'D']

summed_values = sum_values_by_category(values, categories, reference_order)
print(summed_values)'''










