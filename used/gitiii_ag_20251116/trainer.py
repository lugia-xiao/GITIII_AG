import os
import torch
from torch.utils.data import DataLoader,random_split
import pandas as pd
import random
import numpy as np

from gitiii_ag.dataloader import GITIII_dataset
from gitiii_ag.model import GITIII,Loss_function
from gitiii_ag.calculate_PCC import Calculate_PCC

def train_GITIII(target_genes=None,num_neighbors=50,batch_size=256,lr=1e-4,
                 data_dir=None,epochs=50,node_dim=256,num_heads=2,
                 edge_dim=48,att_dim=8,use_cell_type_embedding=True):
    '''

    :param num_neighbors: number of neighboring cells used to predict the cell state of the center cell
    :param batch_size:
    :param lr: learning rate
    :param data_dir: directory of which the preprocessed dataset resides in
    :param epochs: number of training rounds
    :param node_dim: embedding dimension for node in the graph transformer
    :param edge_dim: embedding dimension for edge in the graph transformer
    :param att_dim: dimension needed to calculate for one-head attention
    :return:
    '''
    # Preparation
    if data_dir is None:
        data_dir=os.path.join(os.getcwd(),"data","processed")
    if data_dir[-1]!="/":
        data_dir=data_dir+"/"
    torch.cuda.empty_cache()

    # Load preprocessed dataset
    print("Start loading the dataset")
    dataset = GITIII_dataset(processed_dir=data_dir,num_neighbors=num_neighbors)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    validation_size = total_size - train_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    print("Finish loading the dataset")
    print("Train loader length:", len(train_loader))
    print("Validation loader length:", len(val_loader))

    # Define the model
    ligands_info = torch.load("/".join(data_dir.split("/")[:-2]) + "/ligands.pth")
    genes = torch.load("/".join(data_dir.split("/")[:-2]) + "/genes.pth")

    if target_genes is None:
        target_genes = genes

    print("Target genes:", target_genes)
    torch.save(target_genes,"/".join(data_dir.split("/")[:-2]) + "/target_genes.pth")

    my_model = GITIII(genes=genes, target_genes=target_genes, ligands_info=ligands_info,
                      node_dim=node_dim, edge_dim=edge_dim, num_heads=num_heads,
                      node_dim_small=16, att_dim=att_dim,use_cell_type_embedding=use_cell_type_embedding)
    my_model = my_model.cuda()
    optimizer = torch.optim.AdamW(my_model.parameters(), lr=lr, betas=(0.99, 0.999))
    loss_func = Loss_function(genes, target_genes).cuda()
    evaluator = Calculate_PCC(genes, target_genes)

    records = []
    best_val = 1e10

    potential_saved_name = os.path.join(os.getcwd(),"GITIII.pth")
    if sum([x.find(potential_saved_name) >= 0 for x in os.listdir(".")]) > 0:
        checkpoint = torch.load(potential_saved_name)
        my_model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_val = checkpoint['best_val']
        records = checkpoint['records']

    print("Start training")
    for epochi in range(epochs):
        my_model.train()
        loss_train = 0
        #y_preds_train = []
        #y_train = []
        for (stepi, x) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            x = {k: v.cuda() for k, v in x.items()}
            y_pred = my_model(x)
            y = x["y"]

            if epochi == 0 and stepi==1:
                torch.save(x, "/".join(data_dir.split("/")[:-1]) + "/example_data.pth")

            lossi = loss_func(y_pred, y)
            evaluator.add_input(y_pred, y)

            lossi.backward()
            optimizer.step()

            loss_train = loss_train + lossi.cpu().item()

            if stepi % 500 == 0:
                PCCs = evaluator.calculate_pcc()
                print("Training-> epoch:", epochi, "step:", stepi, "loss:", loss_train / stepi,
                      "median PCC all:", torch.median(PCCs),
                      "max PCC all:", torch.max(PCCs))

        PCC_train = evaluator.calculate_pcc(clear=True)
        print("Finish training, epoch:", epochi)
        print("Loss_all:", loss_train / len(train_loader),
              "median PCC all:", torch.median(PCC_train),
              "max PCC all:", torch.max(PCC_train),)

        checkpoints = {
            'model': my_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'records': records,
            'best_val': best_val
        }
        torch.save(checkpoints, potential_saved_name)

        loss_val = 0
        my_model.eval()
        with torch.no_grad():
            for (stepi, x) in enumerate(val_loader, start=1):
                x = {k: v.cuda() for k, v in x.items()}
                y_pred = my_model(x)
                y = x["y"]

                if stepi <= 10:
                    alphas = torch.abs(y_pred[1][0, :, :])
                    print(torch.topk(torch.mean(alphas / torch.sum(alphas, dim=-2, keepdim=True), dim=-1), k=30, dim=-1))

                evaluator.add_input(y_pred, y)

                lossi = loss_func(y_pred, y)
                loss_val = loss_val + lossi.cpu().item()

                torch.cuda.empty_cache()
                if stepi % 500 == 0:
                    PCCs = evaluator.calculate_pcc()
                    print("Validating-> epoch:", epochi, "step:", stepi, "loss_all:", loss_val / stepi,
                          "median PCC all:", torch.median(PCCs),
                          "max PCC all:", torch.max(PCCs))

        PCC_val= evaluator.calculate_pcc(clear=True)
        print("Finish validating, epoch:", epochi)
        print("Loss_all:", loss_val/ len(val_loader),
              "median PCC all:", torch.median(PCC_val),
              "max PCC all:", torch.max(PCC_val))

        records.append([epochi, loss_train / len(train_loader), PCC_train,
                        loss_val / len(val_loader), PCC_val])
        df = pd.DataFrame(data=records, columns=["epoch", "train_loss", "PCC_train",
                                                 "val_loss", "PCC_val"])
        df.to_csv(os.path.join(os.getcwd(),"record_GITIII.csv"))
        if best_val > loss_val / len(val_loader):
            best_val = loss_val / len(val_loader)
            torch.save(my_model.state_dict(), os.path.join(os.getcwd(),'GITIII_best.pth'))
        print("Best validation loss now:", best_val)

