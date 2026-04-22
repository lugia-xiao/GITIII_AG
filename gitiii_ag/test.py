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

'''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

def plot_boxplot_and_pvalues(data, feature_names):
    """
    Creates a box plot and violin plot for each feature in the data,
    prints a DataFrame of p-values from Mann-Whitney U tests between each pair of columns,
    and prints the p-values between the highest mean column and other columns.

    Parameters:
    data (numpy.ndarray): The input data matrix with shape (n_samples, n_features).
    feature_names (array-like): Array/list of feature names with length n_features.

    Returns:
    pvalues_df (pd.DataFrame): A DataFrame of p-values from pairwise tests.
    pvalues_highest_mean_df (pd.DataFrame): P-values comparing the highest-mean feature to others.
    """
    # Convert feature_names to numpy array for masking
    feature_names = np.array(feature_names)

    # Basic sanity check
    if data.shape[1] != len(feature_names):
        raise ValueError(
            f"Number of feature names ({len(feature_names)}) does not match "
            f"number of columns in data ({data.shape[1]})."
        )

    # ---- Remove 'unassigned' feature(s) if present ----
    mask = feature_names != 'unassigned'
    data = data[:, mask]
    feature_names = feature_names[mask]

    # If everything was unassigned, bail out gracefully
    if data.shape[1] == 0:
        raise ValueError("All features were 'unassigned'; nothing left to plot or test.")

    # ---- Step 1: Create a box plot and violin plot for each feature ----
    num_features = data.shape[1]
    positions = np.arange(1, num_features + 1)

    plt.figure(figsize=(12, 6))

    # Violin plot: one violin per column (feature)
    plt.violinplot(
        data,
        showmeans=False,
        showmedians=True,
        positions=positions
    )

    # Overlay with box plot at the same positions
    plt.boxplot(
        data,
        vert=True,
        patch_artist=True,
        widths=0.15,
        positions=positions
    )

    # Set plot labels and title
    plt.xticks(ticks=positions, labels=feature_names, rotation=45, ha='right')
    plt.xlabel("Feature")
    plt.ylabel("Value")
    plt.title("Box Plot and Violin Plot of Each Feature")
    plt.tight_layout()
    plt.show()

    # ---- Step 2: Calculate p-values for pairwise Mann-Whitney U tests ----
    pvalues_matrix = np.ones((num_features, num_features))  # Initialize with 1s

    for i in range(num_features):
        for j in range(i + 1, num_features):
            _, p_value = mannwhitneyu(
                data[:, i],
                data[:, j],
                alternative='two-sided'
            )
            pvalues_matrix[i, j] = p_value
            pvalues_matrix[j, i] = p_value  # Mirror for symmetry

    pvalues_df = pd.DataFrame(pvalues_matrix, index=feature_names, columns=feature_names)

    print("P-Values for Mann-Whitney U Tests Between Each Pair of Features:")
    print(pvalues_df)

    # ---- Step 3: Highest mean column vs others ----
    mean_values = np.mean(data, axis=0)
    highest_mean_index = np.argmax(mean_values)
    highest_mean_feature = feature_names[highest_mean_index]

    print(f"\nThe column with the highest mean is: {highest_mean_feature}")

    pvalues_highest_mean = {}
    for i in range(num_features):
        if i != highest_mean_index:
            _, p_value = mannwhitneyu(
                data[:, highest_mean_index],
                data[:, i],
                alternative='two-sided'
            )
            pvalues_highest_mean[feature_names[i]] = p_value

    pvalues_highest_mean_df = pd.DataFrame.from_dict(
        pvalues_highest_mean,
        orient='index',
        columns=[f"P-Value with {highest_mean_feature}"]
    )

    print("\nP-Values between the highest mean column and other columns:")
    print(pvalues_highest_mean_df)

    return pvalues_df, pvalues_highest_mean_df



# ----- Test data -----
np.random.seed(42)  # For reproducibility

n_samples = 100
# 5 features with different means
feature_1 = np.random.normal(loc=0.0, scale=1.0, size=n_samples)
feature_2 = np.random.normal(loc=0.5, scale=1.0, size=n_samples)
feature_3 = np.random.normal(loc=1.0, scale=1.0, size=n_samples)
feature_4 = np.random.normal(loc=2.0, scale=1.0, size=n_samples)
feature_5 = np.random.normal(loc=1.5, scale=1.0, size=n_samples)  # will be 'unassigned'

data = np.column_stack([feature_1, feature_2, feature_3, feature_4, feature_5])

feature_names = np.array([
    "feature_A",
    "feature_B",
    "feature_C",
    "feature_D",
    "unassigned"   # this one should be dropped in the function
])

# ----- Run the function -----
pvalues_df, pvalues_highest_mean_df = plot_boxplot_and_pvalues(data, feature_names)'''
'''
from gitiii_ag import data
import importlib.resources as pkg_resources
import torch

def load_dataset(dataset_name='interactions_human'):
    valid_datasets = [
        'interactions_human',
        'interactions_human_nonichenetv2',
        'interactions_mouse',
        'interactions_mouse_nonichenetv2'
    ]

    if dataset_name not in valid_datasets:
        raise ValueError(
            f"Invalid dataset name. Choose from {', '.join(valid_datasets)}"
        )

    # Load the dataset using open_binary
    with pkg_resources.open_binary(data, f'{dataset_name}.pth') as f:
        database = torch.load(f,weights_only=False)
        return database

def rearrange_data(data):
    # Initialize lists to hold all ligands, ligand steps, receptors, and receptor steps
    ligands = []
    ligand_steps = []
    receptors = []
    receptor_steps = []
    pathways=[]
    sources=[]

    # Loop over each item in the input data
    for item in data:
        # Append the ligand to the ligands list
        ligands.append(item[0])
        # Append the ligand steps to the ligand steps list
        ligand_steps.append(item[1])
        # Append the receptor to the receptors list
        receptors.append(item[2])
        # Append the receptor steps to the receptor steps list
        receptor_steps.append(item[3])

        pathways.append(item[5])
        sources.append(item[4])

    # The final output is a list containing four lists: one each for ligands, ligand steps, receptors, and receptor steps
    return [[ligands, ligand_steps], [receptors, receptor_steps], pathways, sources]

def search_interactions_LR_(database,genes,strict=True):
    interactions=[]
    genes_interact=[]
    for i in range(len(database)):
        ligandi=database[i][0]
        receptori=database[i][2]
        flag=None
        if strict:
            flag=(len(set(ligandi).intersection(set(genes)))==len(ligandi) and len(set(receptori).intersection(set(genes)))==len(receptori))
        else:
            flag=(len(set(ligandi).intersection(set(genes)))>0 and len(set(receptori).intersection(set(genes)))>0)
        if flag:
            tmp=list(set(ligandi).intersection(set(genes)))+list(set(receptori).intersection(set(genes)))
            genes_interact=genes_interact+tmp
            interactions.append(database[i])
    genes_interact=list(set(genes_interact))
    interactions=rearrange_data(interactions)
    #print("genes_interact:",len(genes_interact),genes_interact)
    return interactions

print(search_interactions_LR_(load_dataset(),torch.load("../data/genes.pth"),strict=True))
'''

'''import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, linregress

def plot_with_correlations_categorical_colored(x, y, categories, colors, label_x, label_y):
    """
    Scatter plot with PCC, SCC, regression p-value, categorical colors,
    circular points, and no grid.

    Inputs
    ------
    x, y : array-like
        Numeric vectors of the same length.
    categories : array-like
        Vector of strings of the same length as x and y.
    colors : dict
        Mapping from category string -> matplotlib color (e.g., {"A":"red"}).
    label_x, label_y : str
        Axis labels.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    categories = np.asarray(categories)

    if x.shape != y.shape or x.shape != categories.shape:
        raise ValueError("x, y, and categories must have the same shape/length.")

    unique_cats = np.unique(categories)
    missing = [c for c in unique_cats if c not in colors]
    if missing:
        raise KeyError(f"Missing color mapping(s) for categories: {missing}")

    # Correlations (overall)
    pcc, _ = pearsonr(x, y)
    scc, _ = spearmanr(x, y)

    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    # Format p-value
    if p_value == 0:
        p_text = r"$P < 1.00 \times 10^{-324}$"
    else:
        p_sci = f"{p_value:.2e}"
        base, exp = p_sci.split("e")
        base = float(base)
        exp = int(exp)
        p_text = rf"$P = {base:.2f} \times 10^{{{exp}}}$"

    fig, ax = plt.subplots(figsize=(6, 6))

    # Scatter by category
    for cat in unique_cats:
        mask = categories == cat
        ax.scatter(
            x[mask], y[mask],
            s=20, alpha=0.8, marker="o",
            color=colors[cat],
            label=str(cat),
            edgecolors="none"
        )

    # Regression line
    x_vals = np.linspace(x.min(), x.max(), 200)
    y_vals = slope * x_vals + intercept
    ax.plot(x_vals, y_vals, "k-", linewidth=2)

    # Labels and title
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_title(rf"PCC: {pcc:.2f}, SCC: {scc:.2f}, {p_text}")

    ax.grid(False)
    ax.legend(frameon=False, title="Category")
    plt.tight_layout()
    plt.show()

    # Print results
    print(f"PCC: {pcc:.2f}, Spearman: {scc:.2f}")
    n = len(x)
    z_score = scc * np.sqrt((n - 2) / (1 - scc ** 2))
    print("Number of points:", n)
    print("Z-score (from SCC):", z_score)
    print("Regression p-value:", p_value)
    print("-" * 20)

from scipy.stats import pearsonr, spearmanr, linregress
from matplotlib.colors import LinearSegmentedColormap

def plot_with_correlations(x, y, label_x, label_y):
    """
    Scatter plot with PCC, SCC, regression p-value,
    custom colormap, circular points, and no grid.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    # Correlations
    pcc, _ = pearsonr(x, y)
    spearman_corr, _ = spearmanr(x, y)

    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    # Format p-value
    if p_value == 0:
        p_text = r"$P < 1.00 \times 10^{-324}$"
    else:
        p_sci = f"{p_value:.2e}"
        base, exp = p_sci.split("e")
        base = float(base)
        exp = int(exp)
        p_text = rf"$P = {base:.2f} \times 10^{{{exp}}}$"

    # Custom blue → white → red map
    cmap = LinearSegmentedColormap.from_list(
        "blue_white_red", ["#2D3E90", "#FFFFFF", "#9D122A"]
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    # Scatter: circular markers, no grid
    ax.scatter(x, y, c=y, cmap=cmap, s=20, alpha=0.8, marker="o")

    # Regression line
    x_vals = np.linspace(x.min(), x.max(), 200)
    y_vals = slope * x_vals + intercept
    ax.plot(x_vals, y_vals, "k-", linewidth=2)

    # Labels and title
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_title(
        rf"PCC: {pcc:.2f}, SCC: {spearman_corr:.2f}, {p_text}"
    )

    ax.grid(False)  # remove grid
    plt.tight_layout()
    plt.show()

    # Print results
    print(f"PCC: {pcc:.2f}, Spearman: {spearman_corr:.2f}")
    z_score = spearman_corr * np.sqrt((len(x) - 2) / (1 - pcc ** 2))
    print("Number of points:", len(x))
    print("Z-score:", z_score)
    print("Regression p-value:", p_value)
    print("-" * 20)

def signal_strength_all_samples(analyzers,receiver_type,targeted_gene,ligand_gene_list,receptor_gene_list):
    x,y,dementia=[],[],[]
    cnt=0
    for analyzer in analyzers:
        xi,yi=analyzer.visualize_calculate_LR(ligand_genes=ligand_gene_list, receptor_genes=receptor_gene_list, receiver_cell_type=receiver_type,target_gene=targeted_gene,return_y=True)
        x.append(xi)
        y.append(yi)

        dementia=dementia+[cognitive_status[cnt] for j in range(len(yi.flatten()))]
        cnt=cnt+1

    x=np.concatenate(x,axis=0).flatten()
    y = np.concatenate(y,axis=0)
    dementia=np.array(dementia)
    print(x.shape,y.shape,dementia.shape)
    plot_with_correlations_categorical_colored(x,y,dementia,{"Dementia":'red',"No dementia":'grey'},'Signal Strength of '+'-'.join(ligand_gene_list)+'->'+'-'.join(receptor_gene_list),'Measured expression of '+targeted_gene)
    plot_with_correlations(x, y,'Signal Strength of ' + '-'.join(ligand_gene_list) + '->' + '-'.join( receptor_gene_list), 'Measured expression of ' + targeted_gene)
    return x,y,dementia

x,y,dementia=signal_strength_all_samples(analyzers=analyzers,receiver_type='Microglia-PVM',targeted_gene='CD74',ligand_gene_list=['TGFB2'],receptor_gene_list=['TGFBR1','TGFBR2'])

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu


def violin_with_pvalue(
    y,
    categories,
    order=None,
    color="#2E2EFF",
    ylabel="Y",
    title=None,
):
    y = np.asarray(y)
    categories = np.asarray(categories)

    if y.shape != categories.shape:
        raise ValueError("y and categories must have the same length.")

    groups = np.unique(categories) if order is None else np.asarray(order)
    if len(groups) != 2:
        raise ValueError("Exactly two categories are required.")

    g1, g2 = groups
    y1 = y[categories == g1]
    y2 = y[categories == g2]

    # Non-parametric test
    stat, p = mannwhitneyu(y1, y2, alternative="two-sided")

    fig, ax = plt.subplots(figsize=(4, 5))

    # Violin plot
    parts = ax.violinplot(
        [y1, y2],
        positions=[1, 2],
        widths=0.7,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )

    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(1.0)

    # Boxplot inside violin (white)
    ax.boxplot(
        [y1, y2],
        positions=[1, 2],
        widths=0.15,
        patch_artist=True,
        boxprops=dict(facecolor="white", edgecolor="black"),
        medianprops=dict(color="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        showfliers=False,
    )

    # Axis formatting
    ax.set_xticks([1, 2])
    ax.set_xticklabels([str(g1), str(g2)], rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(False)

    if title is not None:
        ax.set_title(title)

    # P-value annotation
    base, exp = f"{p:.2e}".split("e")
    p_text = rf"$P = {float(base):.2f} \times 10^{{{int(exp)}}}$"

    y_max = max(y1.max(), y2.max())
    y_min = min(y1.min(), y2.min())
    y_range = y_max - y_min

    ax.text(
        1.5,
        y_max + 0.15 * y_range,
        p_text,
        ha="center",
        va="bottom",
        fontsize=14,
        style="italic",
    )

    ax.set_ylim(y_min, y_max + 0.3 * y_range)
    plt.tight_layout()
    plt.show()

    print(f"{g1} (n={len(y1)}), {g2} (n={len(y2)})")
    print(f"Mann–Whitney U: {stat:.3g}")
    print(f"P-value: {p:.3e}")


violin_with_pvalue(x,dementia)
violin_with_pvalue(y,dementia)'''

'''import scanpy as sc
import anndata as ad
adatas=[]

for section in sections:
    patient = '.'.join(section.split('.')[:3])
    adatas.append(sc.read_h5ad(f"./raw_data/{patient}.h5ad"))

adatas=ad.concat(adatas)'''

'''import pandas as pd
import torch
import numpy as np
import os

data_dir="./data/processed/"
result_dir='./influence_tensor/'
cell_types=torch.load("./data/processed/cell_types.pth")
cell_type_pair_sequence=[]
for cell_typei in cell_types:
    for cell_typej in cell_types:
        cell_type_pair_sequence.append(cell_typei+"__"+cell_typej)


samples=[]
genes=[]


def calcualte_z_neighbor(x, y, avg_cnti):
    p = torch.mean(y / avg_cnti)
    var = y.shape[0] * p * (1 - p) / 30
    if var == 0:
        return 0
    return float(torch.mean(x - p))


def calculate_strength_spatial_neighbor_adapt(sample):
    # Assuming result_dir is a globally available directory path
    global result_dir, cell_types, cell_type_pair_sequence

    # Load the results
    results = torch.load(result_dir + "edges_" + sample + ".pth")
    cell_type_counts = pd.read_csv("./counts/" + sample + ".csv")
    counts_all = float(np.sum(cell_type_counts.loc[:, "counts"].values))

    # Extract relevant data
    attention_scores = results["attention_score"]  # Shape (B, 49, C)

    proportion = torch.cumsum(results["attention_score"], dim=1) / 8
    y_pred = results["y_pred"].unsqueeze(dim=1)
    attention_scores[torch.abs(proportion) > torch.abs(y_pred) * 0.8] = 0

    proportion = torch.abs(results["attention_score"])
    proportion = proportion / torch.sum(proportion, dim=1, keepdim=True)
    attention_scores[proportion < 0.05] = 0

    expect_cnt_attention_scores = torch.where(attention_scores != 0, torch.ones_like(attention_scores),
                                              torch.zeros_like(attention_scores))

    cell_type_names = np.array(results["cell_type_name"])  # Shape (B, 50)
    true_expression = results["y"]  # Shape (B, C)
    pred_expression = results["y_pred"]

    cell_type_target = [cell_type_names[i][0] for i in range(len(cell_type_names))]
    type_exp_dict = np.load(data_dir + sample + "_TypeExp.npz", allow_pickle=True)
    type_exps = torch.Tensor(np.stack([type_exp_dict[cell_typei] for cell_typei in cell_type_target], axis=0))

    # true_expression=true_expression+type_exps
    # pred_expression=pred_expression+type_exps

    # Initialize a tensor to hold aggregated interaction strengths
    B, _, C = attention_scores.shape
    t = len(cell_types)
    aggregated_interactions = torch.zeros((B, t, C))
    expected_interactions = torch.zeros((B, t, C))

    # Map cell type names to indices
    cell_type_to_index = {ct: idx for idx, ct in enumerate(cell_types)}

    # Aggregate interaction strengths by cell type
    for b in range(B):
        for n in range(1, 50):  # Skip the first element, which is the target cell type
            neighbor_type = cell_type_names[b][n]
            if neighbor_type in cell_type_to_index:
                idx = cell_type_to_index[neighbor_type]
                aggregated_interactions[b, idx] += attention_scores[b, n - 1]
                expected_interactions[b, idx] += expect_cnt_attention_scores[b, n - 1]

    aggregated_interactions1 = torch.abs(aggregated_interactions) / torch.sum(torch.abs(aggregated_interactions), dim=1,
                                                                              keepdim=True)
    aggregated_interactions = torch.where(torch.sum(torch.abs(aggregated_interactions), dim=1, keepdim=True) == 0,
                                          torch.zeros_like(aggregated_interactions), aggregated_interactions1)

    for cell_typei in cell_types:
        mask = (cell_type_names[:, 0] == cell_typei)
        for genei in range(C):
            aggregated_interactions[mask, :, genei] = aggregated_interactions[mask, :, genei] / torch.sum(
                torch.abs(aggregated_interactions[mask, :, genei])) * aggregated_interactions[mask, :, genei].shape[0]

    # Prepare to compute correlations for each cell type pair
    results_matrix = []
    for pair in cell_type_pair_sequence:
        from_type, to_type = pair.split("__")
        if from_type in cell_type_to_index:
            mask = (cell_type_names[:, 0] == to_type)
            filtered_interactions = aggregated_interactions[mask, cell_type_to_index[from_type]]
            filtered_expected_interactions = expected_interactions[mask, cell_type_to_index[from_type]]
            filtered_expressions = true_expression[mask]
            filtered_pred = pred_expression[mask]

            avg_cnt = torch.mean(torch.sum(expected_interactions[mask], dim=1), dim=0)

            if np.sum(mask) == 0:
                results_matrix.append([0 for k in range(C)])
                continue

            # Calculate Pearson correlation coefficient for each gene
            corr_coeffs = []
            for i in range(C):
                gene_interactions = filtered_interactions[:, i]
                gene_expressions = filtered_expressions[:, i]
                expectedi = filtered_expected_interactions[:, i]
                predi = filtered_pred[:, i]
                # r = torch.corrcoef(torch.stack((gene_interactions, gene_expressions)))[0, 1]
                if len(gene_interactions) <= 20 or ((gene_interactions == gene_interactions[0]).all() or (
                        gene_expressions == gene_expressions[0]).all()):
                    corr_coeffs.append(0)
                    continue

                count_from = (cell_type_counts.loc[cell_type_counts["cell_type"] == from_type, "counts"].values)[0]
                count_to = (cell_type_counts.loc[cell_type_counts["cell_type"] == to_type, "counts"].values)[0]
                avg_cnti = avg_cnt[i]
                strength = calcualte_z_neighbor(gene_interactions, expectedi, avg_cnti)

                corr_coeffs.append(float(strength))
            results_matrix.append(corr_coeffs)

    # Convert results to a tensor of shape (t^2, C)
    results_tensor = np.array(results_matrix)
    results_matrix = np.nan_to_num(results_matrix)
    return results_tensor


z_dir = "../network/interaction_strength/"
if not os.path.exists(z_dir):
    os.system("mkdir " + z_dir)

results = []
cnt = 0
for samplei in samples:
    print(cnt + 1, len(samples))
    cnt = cnt + 1
    tmp = calculate_strength_spatial_neighbor_adapt(samplei)
    print(tmp.shape, tmp)
    df = pd.DataFrame(data=tmp, columns=genes, index=cell_type_pair_sequence)
    df.to_csv(z_dir + samplei + ".csv")
'''

