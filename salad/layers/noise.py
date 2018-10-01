import torch
from torch import nn

def get_permutation(n_features, p = .25):

    A = torch.eye(n_features)

    shuffle = torch.rand(n_features) < p

    idc  = torch.arange(n_features).long()
    perm = torch.randperm(shuffle.sum())

    idc_shuffle = idc[shuffle][perm]

    perm = torch.arange(n_features).long()
    
    perm[shuffle] = idc_shuffle
    
    A = torch.eye(n_features)
    A = A[perm,:]
    
    return A
    
    #n = 0
    #for i in idc:
    #
    #    if shuffle[int(i)]:
    #        print(idc_shuffle[n])
    #        n = n + 1
    #    else:
    #        print(i, 'orig')
    

class FeatureRotation(nn.Module):
    
    def __init__(self, n_features, p = .25):
        super().__init__()        
        self.n_features = n_features
        self.p = p
        
    def forward(self, x):
        
        if self.training:
            W = x.new_tensor(get_permutation(self.n_features, self.p))
            x = torch.einsum('ij,nipq->njpq', [W,x])
        
        return x