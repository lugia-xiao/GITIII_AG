import torch
import torch.nn as nn
import numpy as np
import random
import os

from gitiii_ag.attention import GITIII_encoder_last_layer
from gitiii_ag.embedding import Embedding

class GITIII(nn.Module):
    def __init__(self,genes,target_genes,ligands_info,node_dim,edge_dim,
                 num_heads,node_dim_small=16,att_dim=8,
                 use_cell_type_embedding=True):
        super().__init__()
        flag=True
        not_in_genes=[]
        for i in target_genes:
            if i not in genes:
                flag=False
                not_in_genes.append(i)
        assert flag, ",".join(not_in_genes)+" not in the ST measured gene list or modified ST measured gene list"

        self.embeddings=Embedding(genes,ligands_info,node_dim,edge_dim,
                                  use_cell_type_embedding=use_cell_type_embedding,)
        self.last_layer=nn.ModuleList([
            GITIII_encoder_last_layer(node_dim, len(target_genes), edge_dim, node_dim_small, att_dim) for i in range(num_heads)
        ])

    def forward(self,x):
        x=self.embeddings(x)
        outputs,influences=[],[]
        for encoderi in self.last_layer:
            tmp=encoderi(x)
            outputs.append(tmp[0])
            influences.append(tmp[1])
        x=[torch.mean(torch.stack(outputs,dim=-1),dim=-1),torch.mean(torch.stack(influences,dim=-1),dim=-1)]
        return x

class Loss_function(nn.Module):
    def __init__(self,gene_list,target_genes):
        super().__init__()
        self.gene_index = []
        for i in range(len(gene_list)):
            if gene_list[i] in target_genes:
                self.gene_index.append(i)
        self.gene_index = torch.LongTensor(self.gene_index).cuda()
        self.mse=nn.MSELoss()

    def forward(self,y_pred,y):
        output,edges=y_pred
        mse_loss=self.mse(output,y[:,self.gene_index])
        return mse_loss

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    # Python random module
    random.seed(seed_value)
    # Numpy
    np.random.seed(seed_value)
    # PyTorch
    torch.manual_seed(seed_value)
    # CUDA
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU
    # Additional configurations to enforce deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Setting environment variables to further ensure reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed_value)

set_seed(123)

if __name__=="__main__":
    import os

    torch.cuda.empty_cache()
    batch_size = 256
    lr = 1e-4
    folds = 5
    epochs = 200

    data_dir="../data/processed/"
    from torch.utils.data import DataLoader, random_split

    ligands_info = torch.load("/".join(data_dir.split("/")[:-2]) + "/ligands.pth")
    genes = torch.load("/".join(data_dir.split("/")[:-2]) + "/genes.pth")
    target_genes=genes[:10]

    my_model = GITIII(genes,target_genes,ligands_info,node_dim=256,edge_dim=48,num_heads=2,node_dim_small=16,att_dim=8)
    my_model=my_model.cuda()
    optimizer = torch.optim.AdamW(my_model.parameters(), lr=lr, betas=(0.99, 0.999))
    loss_func = Loss_function(gene_list=genes, target_genes=target_genes).cuda()
    import time

    for i in range(20):
        print("start training")
        start = time.time()
        for i in range(10):
            batch={
                "x": torch.randn((batch_size,50,len(genes))),
                "type_exp": torch.randn((batch_size,50,len(genes))),
                "y": torch.randn((batch_size,len(genes))),
                "cell_types": torch.LongTensor([1]).unsqueeze(dim=0).repeat((batch_size,50)),
                "position_x": torch.arange(1,201,4).unsqueeze(dim=0).repeat((batch_size,1)),
                "position_y": torch.arange(1,201,4).unsqueeze(dim=0).repeat((batch_size,1))
            }
            batch = {k: v.cuda() for k, v in batch.items()}
            '''for k, v in batch.items():
                print(k,v.shape,torch.mean(torch.abs(v.float())))'''
            output = my_model(batch)
            print("output",output[0].shape,torch.mean(torch.abs(output[0])),torch.mean(output[0]),torch.mean(torch.abs(batch["y"])),torch.mean(batch["y"]))
            print(loss_func(output, batch["y"]),output[1].shape)
        print(time.time() - start)