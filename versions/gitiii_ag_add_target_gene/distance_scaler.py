import torch
from torch import nn
import monotonicnetworks as lmn

class Monotonic_distance_embedding(nn.Module):
    def __init__(self,out_features,bias=True):
        super().__init__()
        self.out_features=out_features

        lip_nn = nn.Sequential(
            lmn.LipschitzLinear(1, out_features * 4, kind="one-inf",bias=bias),
            lmn.GroupSort(out_features),
            lmn.LipschitzLinear(out_features * 4, out_features*4, kind="inf",bias=bias),
            lmn.GroupSort(out_features),
            lmn.LipschitzLinear(out_features * 4, out_features, kind="inf",bias=bias)
        )
        self.monotonic_nn = lmn.MonotonicWrapper(lip_nn, monotonic_constraints=[[1 for i in range(out_features)] for j in
                                                                           range(1)])

    def forward(self,distance):
        shape=list(distance.shape)
        distances=distance.reshape(-1)

        x=-torch.log(distances+1).unsqueeze(dim=-1)#apply log to avoid nan
        x=self.monotonic_nn(x)
        return x.reshape(shape+[self.out_features])


class Monotonic_distance_embedding1(nn.Module):
    def __init__(self,out_features,bias=True):
        super().__init__()
        self.out_features=out_features
        self.model=nn.Sequential(
            nn.Linear(1,out_features*4),nn.GELU(),nn.Linear(out_features*4,out_features)
        )

    def forward(self,distance):
        shape=list(distance.shape)
        distances=distance.reshape(-1)

        x=-torch.log(distances+1).unsqueeze(dim=-1)#apply log to avoid nan
        x=self.model(x)
        return x.reshape(shape+[self.out_features])

class Monotonic_linear(nn.Module):
    def __init__(self,in_features,out_features,bias=True):
        super().__init__()
        self.out_features=out_features
        self.linear=lmn.MonotonicLayer(in_features, out_features, bias=bias, monotonic_constraints=[[1 for j in range(out_features)] for i in range(in_features)])

    def forward(self,x):
        shape=list(x.shape)[:-1]
        x=x.reshape(-1,x.shape[-1])
        x=self.linear(x)
        x=x.reshape(shape+[self.out_features])
        return x


if __name__=="__main__":
    distance = torch.arange(1, 20).unsqueeze(dim=0).repeat((2, 1))
    print(distance.shape)
    distance_scaler = nn.Sequential(Monotonic_distance_embedding1(out_features=5,bias=True))#,nn.Sigmoid())
    scaler = distance_scaler(distance)

    print(distance)
    print(scaler)

    x=torch.randn((4,1))
    model_linear=Monotonic_linear(1, 3)
    print(x)
    print(model_linear(x))