import torch
import torch.nn as nn
import torch.nn.functional as F

from gitiii_ag.embedding import FFN
from gitiii_ag.dropout_node import ProportionalMasking

def rho(x):
    return torch.sqrt(F.relu(x)) - torch.sqrt(F.relu(-x))

class GITIII_encoder_last_layer(nn.Module):
    def __init__(self, node_dim, in_node, edge_dim, node_dim_small=16, att_dim=8):
        super().__init__()
        self.node_dim_small = node_dim_small
        self.in_node = in_node
        self.edge_dim=edge_dim*2
        self.att_dim=att_dim

        self.W_Q = nn.Linear(node_dim, self.edge_dim * att_dim)
        self.W_K = nn.Linear(node_dim, self.edge_dim * att_dim)
        self.W_Ew = nn.Linear(edge_dim, edge_dim, bias=False)
        self.W_Eb = nn.Linear(edge_dim, edge_dim, bias=False)
        self.W_En = nn.Linear(edge_dim, node_dim)

        self.W_A=FFN(edge_dim,out_dim=in_node,bias=False)

        self.node_transform = nn.Linear(node_dim, in_node)
        self.edge_transform = FFN(edge_dim, out_dim=in_node)
        self.head = FFN(in_node,in_dim=in_node*2, out_dim=in_node)

        self.mask=ProportionalMasking(10)

    def forward(self, x):
        node, edge, embedding1, embedding2 = x
        B, N, C = node.shape

        Q = self.W_Q(node[:,0:1,:]).reshape(B, 1, self.att_dim, self.edge_dim).permute(0,3,1,2)#b,n,h*c->b,n,c,h->b,h,n,c
        K = self.W_K(node).reshape(B, N, self.att_dim, self.edge_dim).permute(0,3,2,1)#b,n,h*c->b,n,c,h->b,h,c,n
        QK = (Q @ K).permute(0, 2, 3, 1)

        edge=edge[:,0:1,:,:]
        #print("abs",torch.mean(torch.abs(rho((QK[:, :, :, :self.edge_dim // 2]) * self.W_Ew(edge)))),torch.mean(torch.abs(QK[:, :, :, self.edge_dim // 2:]*embedding1[:,0:1,:,:])),torch.mean(torch.abs(embedding2[:,0:1,:,:])))
        edge = F.selu(rho((QK[:, :, :, :self.edge_dim // 2]) * self.W_Ew(edge)) + QK[:, :, :, self.edge_dim // 2:]*embedding1[:,0:1,:,:] + embedding2[:,0:1,:,:])  # b,n,n,c_e
        edge = edge[:, 0, 1:, :]

        alphas = F.softmax(self.W_A(edge).permute(0, 2, 1), dim=-1).permute(0, 2, 1)

        edge = self.edge_transform(edge)*alphas
        node = self.node_transform(node[:, 1:, :])*alphas
        tmp = node+edge#self.head(torch.concat([edge, node], dim=-1))

        # random mask
        tmp=self.mask(tmp)

        node = torch.sum(tmp,dim=-2) # b,49,in_node->1
        return [node, tmp]
