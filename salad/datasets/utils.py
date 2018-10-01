import torch

def compute_normalization_stats(data):
    """ Computes Mean and Standard Deviation
    """

    n_samples = 0
    running_mean = None
    running_sd = None

    for ( x, _ ), in data:

        if running_mean is None:
            running_mean = torch.zeros(x.size(1)).double()
            running_sd   = torch.zeros(x.size(1)).double()

            assert x.ndim == 4, "compute_normalization_stats() only implemented for 4d Tensors!"
        
        x_ = x.transpose(1,0).contiguous().view(x.size(1), -1).double()
        
        running_mean += x_.sum(dim=-1)
        running_sd   += x_.var(dim=-1, unbiased=False) * x_.size(1)
        n_samples    += int(x_.size(1))
        
    mu = running_mean / n_samples
    sd = (running_sd  / (n_samples - 1))**.5
    
    return mu, sd