import torch
from sklearn.linear_model import Lasso, lasso_path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import importlib.resources as pkg_resources
from scipy.stats import binned_statistic
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr
from scipy.stats import mannwhitneyu
import pandas as pd

from gitiii_ag import data

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
        database = torch.load(f)
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

def search_interactions_LR(genes,species):
    assert species in ["human", "mouse"], "Species must be one of human or mouse"
    if species=="human":
        database = load_dataset('interactions_human_nonichenetv2')
    elif species=="mouse":
        database = load_dataset('interactions_mouse_nonichenetv2')
    else:
        raise ValueError("Species must be one of human or mouse")
    return search_interactions_LR_(database=database, genes=genes, strict=True)

def group_strings_by_numbers(strings, numbers):
    """
    Group a list of strings based on unique numbers and return as two ordered lists.

    :param strings: List of strings to be grouped.
    :param numbers: List of numbers indicating the grouping.
    :return: Two lists, one with unique numbers and the other with corresponding grouped strings.
    """
    numbers=numbers.copy()[:len(strings)]
    unique_numbers = sorted(set(numbers))
    grouped_strings = [[] for _ in unique_numbers]

    for string, number in zip(strings, numbers):
        index = unique_numbers.index(number)
        grouped_strings[index].append(string)

    return unique_numbers, grouped_strings


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

    indices_to_zero_order = []
    for idx in sorted_indices:
        if idx in ranked_indices.tolist():
            indices_to_zero_order.append(idx)
    '''
    print("Features in order of when their coefficients go to zero:")
    for idx in sorted_indices:
        if zero_order[idx] > 0:
            print(f"{feature_names[idx]} goes to zero at alpha = {alphas_lasso[::-1][zero_order[idx] - 1]:.4f}")
    '''

    return indices_to_zero_order, ranked_indices, coefficients, best_alpha, y_pred_best

def prepare_loess_model(x, y, bins=300, frac=0.013):
    # Determine the range of x and set bin edges
    min_x = np.min(x)
    max_x = np.max(x)
    bin_edges = np.linspace(min_x, max_x, bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[:-1] + bin_width / 2

    # Compute mean y-value for each bin
    bin_means, _, binnumber = binned_statistic(x, y, statistic='mean', bins=bin_edges)

    # Explicitly add the min and max x values to the bin centers and means
    bin_centers = np.append([min_x], bin_centers)
    bin_centers = np.append(bin_centers, [max_x])
    bin_means = np.append([y[x == min_x].mean()], bin_means)
    bin_means = np.append(bin_means, [y[x == max_x].mean()])

    # Calculate the smooth curve using Lowess
    lowess = sm.nonparametric.lowess
    smoothed_data = lowess(bin_means, bin_centers, frac=frac)

    # Plotting for visualization
    plt.figure(figsize=(10, 5))
    plt.scatter(bin_centers, bin_means, alpha=0.5, label='Binned Average')
    plt.plot(smoothed_data[:, 0], smoothed_data[:, 1], 'r-', label='LOESS Smooth')
    plt.title("LOESS Model on Binned Data")
    plt.xlabel("x")
    plt.ylabel("y")
    if np.min(y) >= 0:
        plt.ylim(bottom=0)
    plt.legend()
    plt.grid(True)
    plt.show()

    # Return the fitted LOESS smoothed data for prediction
    return smoothed_data


def predict_with_loess_model(loess_model, new_x):
    # Predict using the fitted LOESS model
    # Find the nearest points and perform linear interpolation
    x_values = loess_model[:, 0]
    y_values = loess_model[:, 1]
    print(x_values[0], x_values[-1])
    if np.min(new_x) < x_values[0] or np.max(new_x) > x_values[-1]:
        raise ValueError("New x value is out of the bounds of the x values in the model.")
    else:
        return np.interp(new_x, x_values, y_values)


def get_distance_scaler(genei, genes, results, attention_scores, distances, receiver_type, count_threshold=100):
    type_filter = np.array(results["cell_type_name"])[:, 0] == receiver_type
    type_filter = torch.Tensor(type_filter).bool()

    # if the number of receiver cell is too small, then use all data for the estimation of distance scaler
    if torch.sum(type_filter) < count_threshold:
        type_filter = torch.ones_like(type_filter)

    if genei in genes:
        scores = torch.abs(attention_scores[:, :, genes.index(genei)][type_filter]).flatten().numpy()
    else:
        print("Input gene not in the measured genes in the spatial transcriptomics data. Using the average of all genes for scaler estimation.")
        attention_scores_avg=torch.abs(attention_scores)
        attention_scores_avg=torch.mean(attention_scores_avg,dim=-1)
        scores = torch.abs(attention_scores_avg[type_filter]).flatten().numpy()

    distances_flatten = distances[type_filter].flatten().numpy()

    model = prepare_loess_model(distances_flatten, scores)

    scalers_value = predict_with_loess_model(model, distances_flatten).reshape(-1, distances.shape[-1])
    scalers = np.zeros_like(distances)

    type_filter = type_filter.numpy().astype(dtype=bool)  # np.array(results["cell_type_name"])[:,0]==receiver_type
    scalers[type_filter, :] = scalers_value
    return scalers


def plot_with_correlations(x, y, label_x, label_y):
    """
    Plots x vs y and displays Pearson and Spearman correlation coefficients in the title.

    Parameters:
    x (array-like): Data for the x-axis.
    y (array-like): Data for the y-axis.
    label_x (str): Label for the x-axis.
    label_y (str): Label for the y-axis.
    """
    # Calculate Pearson and Spearman correlation coefficients
    pcc, _ = pearsonr(x, y)
    spearman_corr, _ = spearmanr(x, y)

    # Plotting the scatter plot
    plt.scatter(x, y)

    # Adding title and labels
    plt.title(f"PCC: {pcc:.2f}, Spearman: {spearman_corr:.2f}")
    plt.xlabel(label_x)
    plt.ylabel(label_y)

    # Display the plot
    plt.show()
    print(f"PCC: {pcc:.2f}, Spearman: {spearman_corr:.2f}",label_x,label_y)

    z_score=spearman_corr*np.sqrt((len(x)-2)/(1-pcc**2))
    print("Number of points:",len(x))
    print("Z-score:",z_score)
    print("-"*20)

def plot_boxplot_and_pvalues(data, feature_names):
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