import torch


def pearson_correlation(y_pred, y):
    """
    Calculates the Pearson Correlation Coefficient for each column in two matrices.

    Args:
    y_pred (torch.Tensor): A 2D tensor representing the predicted values.
    y (torch.Tensor): A 2D tensor representing the actual values.

    Returns:
    torch.Tensor: A 1D tensor containing the PCC for each column.
    """

    # Ensure input tensors are of float type
    y_pred = y_pred.float()
    y = y.float()

    # Calculate mean
    mean_y_pred = torch.mean(y_pred, dim=0)
    mean_y = torch.mean(y, dim=0)

    # Subtract means
    y_pred_minus_mean = y_pred - mean_y_pred
    y_minus_mean = y - mean_y

    # Calculate covariance and variances
    covariance = torch.mean(y_pred_minus_mean * y_minus_mean, dim=0)
    variance_y_pred = torch.mean(y_pred_minus_mean ** 2, dim=0)
    variance_y = torch.mean(y_minus_mean ** 2, dim=0)

    # Calculate PCC
    pcc = covariance / torch.sqrt(variance_y_pred * variance_y)

    return pcc

class Calculate_PCC:
    def __init__(self,gene_list,target_genes):
        self.y_pred=[]
        self.y=[]

        self.gene_index = []
        for i in range(len(gene_list)):
            if gene_list[i] in target_genes:
                self.gene_index.append(i)
        self.gene_index = torch.LongTensor(self.gene_index)

    def add_input(self,y_pred,y):
        y=y.cpu().detach()
        self.y.append(y[:,self.gene_index])
        if len(y_pred)==2:
            y_pred=y_pred[0].cpu().detach()
            self.y_pred.append(y_pred)
        else:
            y_pred = y_pred.cpu().detach()
            self.y_pred.append(y_pred)

    def clear(self):
        self.y_pred = []
        self.y = []

    def calculate_pcc(self,clear=False):
        y=torch.concat(self.y,dim=0)
        y_pred=torch.concat(self.y_pred,dim=0)
        PCC=pearson_correlation(y_pred,y).cpu().detach()
        if clear:
            self.clear()
        return PCC

    def calculate_error(self,clear=True):
        y = torch.concat(self.y, dim=0)
        y_pred = torch.concat(self.y_pred, dim=0)
        if clear:
            self.clear()
        return torch.mean(torch.square(y_pred-y),dim=0)

if __name__=="__main__":
    matrix1=torch.randn((32,100))
    matrix2 = torch.randn((32, 100))
    print(pearson_correlation(matrix1,matrix2).shape)

