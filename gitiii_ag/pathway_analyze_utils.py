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

import cvxpy as cp

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

from scipy.stats import pearsonr, spearmanr, linregress, t

def weighted_linear_regression_with_plots(X, y, weights,
                                          feature_names=None,
                                          alpha=0.05):
    """
    Perform weighted linear regression (WLS) and produce:
      1) Predicted y vs. true y using plot_with_correlations
      2) Coefficient plot with (1-alpha) confidence intervals (no intercept)

    Negative coefficients -> #47BDBE
    Positive/zero coefficients -> #E7796F
    CI plot has no title and no intercept.
    Features are plotted from top (feature 1) to bottom (feature n).
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(weights, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n, c = X.shape
    if y.shape[0] != n or w.shape[0] != n:
        raise ValueError("Shapes of X, y, and weights are inconsistent.")
    if np.any(w < 0):
        raise ValueError("All weights must be non-negative.")

    # Design matrix with intercept
    X_design = np.column_stack([np.ones(n), X])   # (n, c+1)
    p = X_design.shape[1]

    # Weighted normal equation: beta = (X^T W X)^(-1) X^T W y
    W = w[:, None]
    XtWX = X_design.T @ (W * X_design)
    XtWy = X_design.T @ (w * y)
    beta = np.linalg.solve(XtWX, XtWy)

    # Predictions
    y_hat = X_design @ beta

    # Residual variance
    residuals = y - y_hat
    rss = np.sum(w * residuals**2)
    dof = max(n - p, 1)
    sigma2 = rss / dof

    XtWX_inv = np.linalg.inv(XtWX)
    cov_beta = sigma2 * XtWX_inv
    se_beta = np.sqrt(np.diag(cov_beta))

    # Confidence intervals
    t_crit = t.ppf(1 - alpha / 2, dof)
    ci_lower = beta - t_crit * se_beta
    ci_upper = beta + t_crit * se_beta

    # Plot predicted vs observed
    plot_with_correlations(y_hat, y, "Predicted y", "Observed y")

    # ---------------------------------------
    # REMOVE INTERCEPT AND REVERSE ORDER
    # ---------------------------------------
    beta_no_intercept = beta[1:][::-1]         # reverse
    se_no_intercept = se_beta[1:][::-1]
    ci_lower_no_intercept = ci_lower[1:][::-1]
    ci_upper_no_intercept = ci_upper[1:][::-1]

    # Feature names
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(c)]
    if len(feature_names) != c:
        raise ValueError("feature_names length must match number of columns in X.")

    coef_names = feature_names[::-1]  # reverse order

    positions = np.arange(len(beta_no_intercept))

    # ---------------------------------------
    # COEFFICIENT PLOT
    # ---------------------------------------
    fig, ax = plt.subplots(figsize=(7, max(4, 0.4 * len(beta_no_intercept))))

    neg_color = "#47BDBE"
    pos_color = "#E7796F"

    neg_mask = beta_no_intercept < 0
    pos_mask = ~neg_mask

    # Negative coefficients
    if np.any(neg_mask):
        ax.errorbar(
            beta_no_intercept[neg_mask],
            positions[neg_mask],
            xerr=t_crit * se_no_intercept[neg_mask],
            fmt="o",
            capsize=4,
            linewidth=1,
            color=neg_color,
        )

    # Positive coefficients
    if np.any(pos_mask):
        ax.errorbar(
            beta_no_intercept[pos_mask],
            positions[pos_mask],
            xerr=t_crit * se_no_intercept[pos_mask],
            fmt="o",
            capsize=4,
            linewidth=1,
            color=pos_color,
        )

    # Zero line
    ax.axvline(0.0, linestyle="--", color="black")

    ax.set_yticks(positions)
    ax.set_yticklabels(coef_names)
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("")  # no label
    # no title

    plt.tight_layout()
    plt.show()

    # Return everything (including intercept)
    results = {
        "beta": beta,
        "se": se_beta,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "y_hat": y_hat,
        "residuals": residuals,
        "sigma2": sigma2,
        "dof": dof,
    }

    print(results)
    return results


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, lasso_path
from sklearn.model_selection import KFold

def perform_lasso_cv_with_mse(
    X,
    y,
    feature_names,
    weights=None,        # sample weights, same shape as y
    mark_topk=5,
    cv=5,
    max_iter=20000,
    visualize=True
):
    """
    Perform (optionally weighted) LASSO regression with custom cross-validation
    to select the best alpha based on MSE.

    Parameters:
    - X: numpy array of shape (n, c), data matrix
    - y: numpy array of shape (n,), response vector
    - feature_names: list of str, names of the features
    - weights: numpy array of shape (n,), non-negative weights (>=0).
               If None, all weights are treated as 1.
    - mark_topk: int, number of top coefficients to annotate on the plot (default: 5)
    - cv: int, number of cross-validation folds (default: 5)
    - max_iter: int, maximum iterations for LASSO (default: 20000)
    - visualize: bool, whether to show plots

    Returns:
    - indices_to_zero_order: list, indices of selected features ordered by when
      their coefficients go to zero along the path
    - ranked_indices: numpy array, indices of selected features ranked by
      absolute value of coefficients at best alpha
    - coefficients: numpy array, LASSO coefficients of all features (including zeros)
    - best_alpha: float, best alpha value found using (weighted) MSE
    - y_pred_best + y_mean: numpy array, predictions on X in the original y scale
    """

    # -------------------- CLEAN INPUTS --------------------
    X = np.nan_to_num(X.copy())
    y = np.nan_to_num(y.copy())

    n_samples = y.shape[0]

    if weights is None:
        weights = np.ones(n_samples, dtype=float)
    else:
        weights = np.nan_to_num(np.asarray(weights, dtype=float))
        if weights.shape != y.shape:
            raise ValueError("weights must have the same shape as y.")
        if np.any(weights < 0):
            raise ValueError("All weights must be >= 0.")

    # Unweighted normalization as requested
    y_mean = np.mean(y)
    y = y - y_mean

    # Standardize features (unweighted)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -------------------- DEFINE ALPHAS --------------------
    alphas = np.logspace(-4, 0, 50)

    # -------------------- CROSS-VALIDATION (WEIGHTED LOSS) --------------------
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    mse_means = []

    for alpha in alphas:
        mse_folds = []
        for train_index, val_index in kf.split(X_scaled):
            X_train_fold, X_val_fold = X_scaled[train_index], X_scaled[val_index]
            y_train_fold, y_val_fold = y[train_index], y[val_index]
            w_train_fold, w_val_fold = weights[train_index], weights[val_index]

            # Use sklearn's built-in sample_weight here
            lasso = Lasso(alpha=alpha, max_iter=max_iter)
            lasso.fit(X_train_fold, y_train_fold, sample_weight=w_train_fold)

            # Predict on validation data
            y_val_pred = lasso.predict(X_val_fold)

            # Weighted MSE on validation fold
            mse_fold = np.average((y_val_fold - y_val_pred) ** 2, weights=w_val_fold)
            mse_folds.append(mse_fold)

        mse_means.append(np.mean(mse_folds))

    mse_means = np.array(mse_means)
    best_alpha_index = np.argmin(mse_means)
    best_alpha = alphas[best_alpha_index]
    print("Best alpha based on (weighted) MSE:", best_alpha)

    # -------------------- FINAL WEIGHTED LASSO FIT --------------------
    lasso_best = Lasso(alpha=best_alpha, max_iter=max_iter)
    lasso_best.fit(X_scaled, y, sample_weight=weights)

    # Predictions on original (unweighted) X_scaled
    y_pred_best = lasso_best.predict(X_scaled)

    # Coefficients
    coefficients = lasso_best.coef_

    # -------------------- RANK NON-ZERO COEFFICIENTS --------------------
    non_zero_indices = np.where(coefficients != 0)[0]
    non_zero_coefficients = coefficients[non_zero_indices]
    if len(non_zero_indices) > 0:
        ranked_indices = non_zero_indices[np.argsort(-np.abs(non_zero_coefficients))]
    else:
        ranked_indices = np.array([], dtype=int)

    # Top-k for annotation
    mark_topk = min(mark_topk, len(ranked_indices))
    top_k_indices = ranked_indices[:mark_topk]
    top_k_coefficients = coefficients[top_k_indices]
    top_k_feature_names = [feature_names[i] for i in top_k_indices]

    # -------------------- COEFFICIENT PATH (WEIGHTED DATA FOR PATH ONLY) --------------------
    # lasso_path has no sample_weight argument, so use sqrt(weights) trick *only* here
    sw_all = np.sqrt(weights)
    X_w = X_scaled * sw_all[:, None]
    y_w = y * sw_all

    alphas_lasso, coefs_lasso, _ = lasso_path(
        X_w, y_w, alphas=alphas, max_iter=5000
    )

    # -------------------- PLOTTING --------------------
    if visualize:
        # ---- Figure 1: (Weighted) MSE vs Alpha ----
        plt.figure(figsize=(7, 5))
        plt.plot(alphas, mse_means)
        plt.axvline(best_alpha, linestyle='--', color='r')
        plt.scatter(best_alpha, mse_means[best_alpha_index], color='r')
        plt.xscale('log')
        plt.xlabel('Alpha')
        plt.ylabel('Weighted Mean Squared Error')
        plt.title('Weighted MSE vs Alpha')
        plt.tight_layout()
        plt.show()

        # ---- Figure 2: Coefficient Paths ----
        plt.figure(figsize=(7, 5))
        plt.plot(alphas_lasso, coefs_lasso.T)
        plt.axvline(best_alpha, linestyle='--', color='r', label='Best alpha')
        plt.xscale('log')
        plt.xlabel('Alpha')
        plt.ylabel('Coefficient Value')
        plt.title('Coefficient Paths (Weighted LASSO)')

        # Annotate top k predictors at the best alpha
        for i, (index, coef) in enumerate(zip(top_k_indices, top_k_coefficients)):
            plt.annotate(
                top_k_feature_names[i],
                xy=(best_alpha, coef),
                xytext=(5, 5),
                textcoords='offset points',
                arrowprops=dict(facecolor='black', arrowstyle='->')
            )

        plt.legend()
        plt.tight_layout()
        plt.show()

        # ---- Figure 3: Top 10 |Coefficients| at Best Alpha ----
        if len(ranked_indices) > 0:
            top_n_bar = min(10, len(ranked_indices))
            bar_indices = ranked_indices[:top_n_bar]
            bar_coefficients = coefficients[bar_indices]
            bar_feature_names = [feature_names[i] for i in bar_indices]

            colors = ['#E7796F' if c > 0 else '#47BDBE' for c in bar_coefficients]

            plt.figure(figsize=(7, 5))
            y_pos = np.arange(top_n_bar)
            plt.barh(y_pos, bar_coefficients, color=colors)
            plt.yticks(y_pos, bar_feature_names)
            plt.axvline(0, color='k', linewidth=0.8)
            plt.xlabel('Coefficient Value')
            plt.title('Top 10 |Coefficients| at Best Alpha (Weighted)')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()

        if len(ranked_indices) > 0:
            top_n_bar = min(10, len(ranked_indices))
            bar_indices = ranked_indices[:top_n_bar]
            bar_feature_names = [feature_names[i] for i in bar_indices]
            #bar_coefficients = coefficients[bar_indices]
            weighted_linear_regression_with_plots(X=X_scaled[:,[feature_names.index(i) for i in bar_feature_names]],y=y,weights=weights,feature_names=bar_feature_names)


    # -------------------- ZERO-ORDER RANKING --------------------
    # coefs_lasso: shape (n_features, n_alphas)
    zero_order = np.sum(coefs_lasso != 0, axis=1)
    sorted_indices = np.argsort(zero_order)[::-1]

    indices_to_zero_order = []
    ranked_list = ranked_indices.tolist()
    for idx in sorted_indices:
        if idx in ranked_list:
            indices_to_zero_order.append(idx)

    # y_pred_best is on centered scale; add back unweighted mean
    return indices_to_zero_order, ranked_indices, coefficients, best_alpha, y_pred_best + y_mean



# ----- Core monotone spline fitter (piecewise-linear on knots) -----

def _build_linear_spline_design(x, knots):
    x = np.asarray(x)
    knots = np.asarray(knots)
    n = x.shape[0]
    m = knots.shape[0]
    A = np.zeros((n, m), dtype=float)

    idx = np.searchsorted(knots, x, side="right") - 1
    idx = np.clip(idx, 0, m - 2)

    t_left = knots[idx]
    t_right = knots[idx + 1]
    denom = (t_right - t_left)
    denom[denom == 0] = 1.0

    w_right = (x - t_left) / denom
    w_left = 1.0 - w_right

    rows = np.arange(n)
    A[rows, idx] += w_left
    A[rows, idx + 1] += w_right

    return A


def prepare_spline_model(
    x,
    y,
    n_knots=40,
    n_grid=300,
    lambda_smooth=1e-3,           # slightly smaller default
    small_x_quantile=0.2,         # upweight first 20% of distances
    small_x_weight_factor=5.0,    # how much more we care about small x
    visualize=True,
):
    """
    Smooth, monotone-decreasing, non-negative spline f(x) for strength vs distance.

    Changes vs previous version:
      - knots are taken at quantiles of x (more flexibility where data are dense,
        often small x).
      - small distances are upweighted in the loss via small_x_weight_factor.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same length.")

    # Sort by x
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    y_sorted = np.maximum(y_sorted, 0.0)

    n = x_sorted.shape[0]

    # ---- Quantile-based knots (more knots where data is dense) ----
    quantiles = np.linspace(0.0, 1.0, n_knots)
    knots = np.quantile(x_sorted, quantiles)

    # Make sure knots are strictly increasing (tiny jitter if needed)
    knots = np.unique(knots)
    if knots.shape[0] < 3:
        raise ValueError("Not enough distinct knot positions; check your x data.")
    # Recompute n_knots after uniqueness
    n_knots_eff = knots.shape[0]

    A = _build_linear_spline_design(x_sorted, knots)

    # Smoothness matrix
    D = np.zeros((n_knots_eff - 2, n_knots_eff))
    for i in range(n_knots_eff - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0

    # ---- Weights: emphasize small x if desired ----
    w = np.ones(n, dtype=float)
    if small_x_weight_factor is not None and small_x_weight_factor > 1.0:
        x_thresh = np.quantile(x_sorted, small_x_quantile)
        w[x_sorted <= x_thresh] *= small_x_weight_factor

    sqrt_w = np.sqrt(w)

    beta = cp.Variable(n_knots_eff)

    # Weighted data fit + smoothness
    resid = cp.multiply(sqrt_w, A @ beta - y_sorted)
    data_fit = (1.0 / n) * cp.sum_squares(resid)
    smooth_penalty = lambda_smooth * cp.sum_squares(D @ beta)
    objective = cp.Minimize(data_fit + smooth_penalty)

    constraints = [beta >= 0]
    for j in range(n_knots_eff - 1):
        constraints.append(beta[j] >= beta[j + 1])

    problem = cp.Problem(objective, constraints)
    problem.solve()

    if beta.value is None:
        raise RuntimeError("CVXPY failed to find a solution for the spline model.")

    beta_hat = np.array(beta.value, dtype=float)

    # Sample model on a grid
    x_min, x_max = x_sorted[0], x_sorted[-1]
    x_grid = np.linspace(x_min, x_max, n_grid)
    y_grid = np.interp(x_grid, knots, beta_hat)
    y_grid = np.maximum(y_grid, 0.0)

    model_data = np.column_stack([x_grid, y_grid])

    if visualize:
        plt.figure(figsize=(10, 5))
        plt.plot(model_data[:, 0], model_data[:, 1], "r-", label="Monotone spline fit")
        plt.title("Monotone Constrained Spline (strength vs distance)")
        plt.xlabel("Distance")
        plt.ylabel("Interaction strength")
        if np.min(y_sorted) >= 0:
            plt.ylim(bottom=0)
        plt.legend()
        #plt.grid(True)
        plt.show()

    return model_data


def predict_with_spline_model(spline_model, new_x):
    x_values = spline_model[:, 0]
    y_values = spline_model[:, 1]
    new_x = np.asarray(new_x, dtype=float)

    if np.min(new_x) < x_values[0] or np.max(new_x) > x_values[-1]:
        raise ValueError("New x value is out of the bounds of the x values in the model.")
    return np.interp(new_x, x_values, y_values)


# ----- Plug into your distance-scaler pipeline -----

def get_distance_scaler(
    genei,
    genes,
    results,
    attention_scores,
    distances,
    receiver_type,
    count_threshold=100,
    visualize=True,
    n_knots=40,
    n_grid=300,
    lambda_smooth=1e-2,
):
    """
    Estimate a distance-based scaler for cell–cell interaction strength
    using a smooth, monotone-decreasing, non-negative spline model.

    Returns:
      scalers: array with the same shape as `distances`.
    """
    # Filter receiver type
    type_filter = np.array(results["cell_type_name"])[:, 0] == receiver_type
    type_filter = torch.Tensor(type_filter).bool()

    # If too few receiver cells, use all data
    if torch.sum(type_filter) < count_threshold:
        type_filter = torch.ones_like(type_filter)

    # Extract scores
    if genei in genes:
        scores = torch.abs(
            attention_scores[:, :, genes.index(genei)][type_filter]
        ).flatten().numpy()
    else:
        print(
            "Input gene not in the measured genes in the spatial transcriptomics data. "
            "Using the average of all genes for scaler estimation."
        )
        attention_scores_avg = torch.abs(attention_scores)
        attention_scores_avg = torch.mean(attention_scores_avg, dim=-1)
        scores = torch.abs(attention_scores_avg[type_filter]).flatten().numpy()

    # Distances for the same cells
    distances_flatten = distances[type_filter].flatten().numpy()

    # Fit monotone constrained spline model
    model = prepare_spline_model(
        distances_flatten,
        scores,
        n_knots=n_knots,
        n_grid=n_grid,
        lambda_smooth=lambda_smooth,
        visualize=visualize,
    )

    # Predict scalers at those distances (same shape logic as before)
    scalers_value = predict_with_spline_model(
        model, distances_flatten
    ).reshape(-1, distances.shape[-1])

    scalers = np.zeros_like(distances)
    type_filter_np = type_filter.numpy().astype(bool)
    scalers[type_filter_np, :] = scalers_value

    return scalers


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

def plot_boxplot_and_pvalues(data, feature_names):
    """
    Creates a box plot for each feature in the data (without outlier dots),
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

    # ---- Step 1: Create a box plot for each feature (no outlier dots) ----
    num_features = data.shape[1]
    positions = np.arange(1, num_features + 1)

    plt.figure(figsize=(12, 6))

    # Box plot only, with outliers (fliers) hidden
    plt.boxplot(
        data,
        vert=True,
        patch_artist=True,
        widths=0.5,
        positions=positions,
        showfliers=False  # <-- hides the outlier dots
    )

    # Set plot labels and title
    plt.xticks(ticks=positions, labels=feature_names, rotation=45, ha='right')
    plt.xlabel("Feature")
    plt.ylabel("Value")
    plt.title("Box Plot of Each Feature (Outliers Hidden)")
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