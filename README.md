# **Linking target gene with ligand-receptor interactions via interpreting graph transformer–learned spatial cell state–niche correlations**

## Description

GITIII-AG is an user-friendly python packaged used for analyzing cell-cell interaction (CCI) using single-cell-level spatial transcriptomics to infer **which ligand-receptor signaling pathway influence the gene of interest in the cell type of interest**. 



## Installation

It is recommanded to install it in an environment with pytorch installed.

```bash
git clone https://github.com/lugia-xiao/GITIII_AG.git
cd GITIII_AG
pip install .
```

## Quick start

### Data you need:

1. A single cell level spatial transcriptomics dataset (converted to the .csv format, easier for cross-platform)

   :param `df_path`: str, the path of your dataset, which should be a .csv file with columns of:

   - genes (more than one column), as described below, these columns form the (normalized) expression matrix.
         values in these columns must be int or float
   - "centerx": x coordinates of the cells. int or float
   - "centery": y coordinates of the cells. int or float
   - "section": which tissue slide this cell belongs to, since a dataset may contain more than one slide. string
   - "subclass": the cell type annotation for this cell. string

2. :param `genes`: list of str, a python list of measured gene names in the dataset

3. A scRNA-seq or snRNA-seq dataset (raw counts, unnormalized, with .obs['celltype'] of cell type annotations (no cell type annotation is also ok)) if your single cell level spatial transcriptomics dataset does not measure all genes. It is used for imputed not measured genes' expression. **Note: if you already have a single cell level spatial transcriptomics dataset with full gene coverage (number of genes > 10000), then you don't need this and the algorithm will automatically ignore the imputation step.**

```python
# Suppose that your data path is:
df_path="./raw_data/BC.csv"
genes=[...]

import scanpy as sc
sc_adata=sc.read_h5ad("./raw_data/BC_sc.h5ad") # a raw count adata
```

### Import necessary packages and fix random seeds

```python
import torch
import numpy as np
import random
import os
import scanpy as sc

def set_random_seed(seed: int):
    # Set seed for Python's built-in random module
    random.seed(seed)
    
    # Set seed for NumPy
    np.random.seed(seed)
    
    # Set seed for PyTorch (both CPU and GPU)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU.
    
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Optionally set the seed for Scanpy (if any stochastic behavior occurs)
    sc.settings.n_jobs = 1  # To avoid parallel execution randomness
    
    # Set the environment variable for deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"Random seed set to {seed} for reproducibility.")

# Example usage:
set_random_seed(42)

# Import necessary python packages
import gitiii_ag
```

### Train GITIII-AG model and estimate the influence tensor

#### Function manual

##### Necessary inputs or necessary parameters to determine

:param `df_path`: str, the path of your dataset, which should be a .csv file with columns of:
- genes (more than one column), as described below, these columns form the expression matrix. Values in these columns must be int or float
- "centerx": x coordinates of the cells. int or float
- "centery": y coordinates of the cells. int or float
- "section": which tissue slide this cell belongs to, since a dataset may contain more than one slide. string
- "subclass": the cell type annotation for this cell. string
  

:param `genes`: list of str, a python list of measured gene names in the dataset

:param `target_genes`: list of str, a python list of target gene names in the dataset, if None, set to all measured genes in the image-based ST dataset

:param `species`: str, the species name, must be 'human' or 'mouse'

:param `library_size_normalize`: bool, whether to do library size normalization to `target_sum`

:param `target_sum`: float, target sum of library size normalization.

:param `use_log_normalize`: bool, whether to use log normalization to transform the data in the ST dataset (log(x+1))

---

---

---

##### Default hyperparameters (You can leave them not input)

:param use_nichenetv2: bool, whether to use nichenetv2 to find ligand genes

:param visualize_when_preprocessing: bool, whether to visualize the ST dataset when preprocessing

:param distance_threshold: float or int, if the distance between one cell and its nearest neighbor is
            above this threshold, then we think this cell is moved during the preparation of the tissue slide in
            the wet lab experiment, we would not include this cell in the analysis

:param process_num_neighbors: int, how many k-nearest neighbor are needed to be calculated in preprocessing

:param num_neighbors: int, number of neighboring cells used to predict the cell state of the center cell

:param batch_size_train: int, batch size at the training step

:param lr: learning rate, float

:param epochs: int, number of training rounds

:param node_dim: int, embedding dimension for node in the graph transformer

:param edge_dim: int, embedding dimension for edge in the graph transformer

:param att_dim: int, dimension needed to calculate for one-head attention

:param use_cell_type_embedding: bool, whether to use cell type embedding

:param num_heads: int, number of attention heads

:param batch_size_val: int, batch size at the evaluating step

```python
target_genes=['MRC1']

estimator=gitiii_ag.estimator.GITIII_estimator(df_path=df_path,genes=genes,target_genes=target_genes, use_log_normalize=False, library_size_normalize=False,species="human",use_nichenetv2=True,visualize_when_preprocessing=False,distance_threshold=80,process_num_neighbors=50,num_neighbors=50,batch_size_train=256,lr=1e-4,epochs=50,node_dim=256,edge_dim=48,att_dim=8,batch_size_val=256,use_cell_type_embedding=True)

estimator.preprocess_dataset()
estimator.train()
estimator.calculate_influence_tensor()
```

### Init pathway analyzer

:param sample: str, the name of the sample you are going to analyze (remind that there is one column section in the input ST dataset csv)

:param sc_adata, the scRNA-seq dataset.

`st_label` and `sc_label` are cell type annotation vector, of the same shape as `df.shape[0]` and `sc_adata.shape[0]

**Note:**

`genes_to_remove_from_LR`: A list of gene names that to be excluded from analysis, here since MRC1 is a receptor gene, we don't want to predicted MRC1 given MRC1, so we remove any ligand receptor pathways related to the MRC1 gene in the pathway analysis. But if the gene you are to analyze is not a receptor gene, then no need to fill in this input.

```python
sample='sample01'
sc.pp.normalize_total(sc_adata, target_sum=1e4)
sc.pp.log1p(sc_adata)

import pandas as pd
df=pd.read_csv("./data/processed/sample01.csv")
#df=df[df['flag']!=0]
st_label=np.array(df[df["section"]==sample]['subclass_original'])
print(np.unique(st_label))

pathway_analyzer=gitiii_ag.pathway_analyzer.Pathway_analyzer(sc_adata=sc_adata,st_sample=sample,species="human",sc_label=sc_adata.obs["celltype"],st_label=st_label,genes_to_remove_from_LR=['MRC1'])
```

### Impute full transcriptome with maxfuse

For easier use, we integrated the maxfuse (A great method to impute full transcriptome in imaging-based ST using scRNA-seq) to the GITIII-AG package so you can use it just with one line.

For choice of hyperparameters, please refer to the maxfuse paper (https://www.nature.com/articles/s41587-023-01935-0). You can also choose other methods as described later in the Quick Q&A section.

`common_genes`: The common genes between ST and scRNA-seq dataset, if None, the algorithm will automatically detect.

`to_csv_path`: file path to save the results of maxfuse.

```python
pathway_analyzer.maxfuse_integrate(common_genes=None,to_csv_path=os.path.join(os.getcwd(), "data", sample+"_match.csv"),visualize=True,label_path=os.path.join(os.getcwd(),"data",sample+"_labels.csv"),scale_normalize_sc=False,use_my_debugged_function_of_wrong_maxfuse_code=False,propagate_wt1=0.4, propagate_wt2=0.4)
```

### Run the GITIII-AG algorithm

`receiver_type`: receiver cell type of interest

`targeted_gene`: target gene of interest

`visualize`: whether to do visualization

```python
targeted_gene='MRC1'

significant_LR1,significant_LR1_to_zero_order=pathway_analyzer.find_significant_LR_with_known_receiver_value__scaled_LR_VS_cell_state(receiver_type=receiver_type,targeted_gene=targeted_gene,visualize=True)
```

`significant_LR1` is a ordered list of ligand-receptor signaling pathways of the LR with the largest coefficients that influence the target gene of interest in the receiver cell type in the LASSO analysis,``significant_LR1_to_zero_order` is another ordered list but ordered by which variable's coefficient go to 0 latest when increasing the penalty in LASSO.

### Visualize the results

#### Visualize one specific pathway

```python
# Transform the signal strength for a better visualization

def f(x):
    return 1-(1/((x*3+1)))

import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'jet'
pathway_analyzer.visualize_calculate_LR(ligand_genes=['CSF1'],receptor_genes=['CSF1R'],receiver_cell_type='Macrophage',target_gene='MRC1',normalize=f)
```

#### Visualize the top pathways that influence MRC1 in macrophages

```python
print(significant_LR1)
for i in range(5):
    if i==len(significant_LR1)-1 or len(significant_LR1)-1<0:
        break
    LR_pairi=significant_LR1[i]
    print("Now showing pathway",LR_pairi)
    ligand_gene=LR_pairi.split("->")[0]
    receptor_gene=LR_pairi.split("->")[1]
    pathway_analyzer.visualize_LR(ligand_gene=ligand_gene,receptor_gene=receptor_gene,receiver_cell_type=receiver_type,target_gene=targeted_gene)
    print("="*20)
```

#### Identify sender cell types of one specific pathway

```python
pathway_analyzer.identify_source_sender_type__known_LR_and_receiver_type(receiver_type='Macrophage',LR='CSF1->CSF1R',target_gene='MRC1')
```

### Jointly analyze multiple ST slides

You can jointly analyze multiple ST slides by:

```python
# Import necessary python packages
import gitiii_ag

# multiple ST slides to analysis, remind the 'section' column in the input ST dataset .csv file
sections=[
    "H20.33.001.CX28.MTG.02.007.1.02.03",
    "H21.33.001.Cx22.MTG.02.007.1.01.03",
    "H21.33.031.CX24.MTG.02.007.1.01.01",
    "H20.33.040.Cx25.MTG.02.007.1.01.04",
    "H21.33.015.Cx26.MTG.02.007.1.1",
    "H21.33.040.Cx22.MTG.02.007.3.03.04",
    "H21.33.023.Cx26.MTG.02.007.1.03.04",
    "H20.33.001.Cx28.MTG.02.007.1.01.03"
]

_adata_cache={}
_match_df_cache={}
```



```python
def build_pathway_analyzer_multi_sample(sections, genes_to_remove=None):
    analyzers=[]
    for section in sections:
        patient='.'.join(section.split('.')[:3])
        if patient not in _adata_cache:
            _adata_cache[patient]=sc.read_h5ad(f"./raw_data/{patient}.h5ad")
        sc_adata=_adata_cache[patient].copy()
        kwargs=dict(
            sc_adata=sc_adata,
            st_sample=section,
            species="human",
            sc_label=None,
            st_label=None,
            filter_noise_proportion=0.02,
            discard_no_match_threshold=0.7,
            num_neighbors=50,
        )
        if genes_to_remove is not None:
            kwargs["genes_to_remove_from_LR"]=genes_to_remove
        pathway_analyzer=gitiii_ag.pathway_analyzer.Pathway_analyzer(**kwargs)
        if section not in _match_df_cache:
            _match_df_cache[section]=pd.read_csv(f"./data/{section}_match.csv")
        pathway_analyzer.integrate_sc_st_with_match_df(match_df=_match_df_cache[section].copy(),check=False)
        analyzers.append(pathway_analyzer)
    return gitiii_ag.pathway_analyzer_multi_samples.Pathway_analyzer_multi_samples(analyzers)

pathway_analyzer_multi_sample=build_pathway_analyzer_multi_sample(sections)

targeted_gene='CD74'

# Build a pathway analyzer with the receptor removed from LR pathways for this gene
current_analyzer=build_pathway_analyzer_multi_sample(sections, genes_to_remove=[targeted_gene])
```



```python
significant_LR1,significant_LR1_to_zero_order=pathway_analyzer.find_significant_LR_with_known_receiver_value__scaled_LR_VS_cell_state(receiver_type=receiver_type,targeted_gene=targeted_gene,visualize=True)
```

## Detailed tutorials

In ./tutorials/breast_cancer_macrophage_MRC1.ipynb



## Quick Q&A

1. About imputing the full transcriptome on the imaging-based ST with scRNA-seq:

   1. If you prefer using **imputation methods** other than the built-in maxfuse?

      You can simply transform your imputed dataset into csv dataset form as described above, then feed them into the GITIII-AG pipeline. The package will detect your input number of genes, if number of genes in the ST dataset is >10000, then GITIII-AG will automatically skip the imputation step.

   2. If I am using the latest spatial transcriptomics technique (e.g. have full transcriptome with spatial single cell resolution)?

      You can simply skip the imputation step and the package will run as normal.



### Acknowledgement

This project is built on these excellent projects:

- https://github.com/shuxiaoc/maxfuse



and my previous project

- https://github.com/lugia-xiao/GITIII



1