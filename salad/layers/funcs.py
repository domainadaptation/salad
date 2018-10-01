import torch

def concat(x, z):

    """ Concat 4D tensor with expanded 2D tensor
    """

    _,_,h,w = x.size()
    n,d     = z.size()
    z_ = z.view(n,d,1,1).expand(-1,-1,h,w)

    return torch.cat([x, z_], dim=1)