""" Metrics and Divergences for Correlation Matrices
"""

import torch

#def cov(x):
#    n, d  = x.size()
#
#    xm = x - x.mean(dim = 0, keepdim=True)
#    C = 1. / (n - 1) * torch.mm(xm.transpose(1,0), x)
#
#    return C 

def cov(x, eps = 1e-5):
    """ Estimate the covariance matrix 
    """
    assert len(x.size()) == 2, x.size()
    
    N, d = x.size()
    
    reg = eps * x.new_tensor(torch.eye(d))
    
    x_ = x - x.mean(dim=0, keepdim = True)
    return torch.einsum('ni,nj->ij',(x_,x_)) / (N - 1) + reg

#def logeig(M, eps = 1e-8):
#    """ Compute log transform on the eigenvalues of a matrix
#    """ 
#
#    u,s,vh = torch.svd(M, some=True)
#    Mlog = (u * torch.log(eps + s)).mm(vh.transpose(1,0))
#
#    return Mlog

def stable_logdet(A):
    """ Compute the logarithm of the determinant of matrix in a numerically stable way
    """

    G = torch.potrf(A, upper=False)
    return 2*torch.log(torch.diag(G)).sum()
    
    #N = len(A)
   # 
    #with torch.no_grad():
    #    mu = abs(A).max()
    
    #scaledA = A / mu    
    
    #log_det_scale =  N * torch.log(mu)
    
    #detA    = torch.det(scaledA)
    
    #logdetA = torch.log(detA) + log_det_scale
    
    #return logdetA


def getdata(N,d,std):
    x = torch.randn(N,d)
    y = std * torch.randn(N,d)
    
    C = cov(x)
    D = cov(y)
    
    return C, D

def apply(C, func):
    e, v = torch.eig(C, eigenvectors=True)
    reg = 0
    with torch.no_grad():
        reg = 1e-5 -  e[:,0].min()
    e = func(e[:,0] + reg)
    C_ = torch.einsum('id,d,jd->ij', (v, e, v))
    return C_

def logeuclid(A, B):

    def log_mat(X):
        N, d = X.size()
        u,s,vh = torch.svd(X)
        eigXtX = (s**2) / (N - 1)
        logeig = torch.log(eigXtX)
        #logX =   torch.einsum('id,d,jd->ij', (vh,logeig,vh))
        logX = vh.mm(torch.diag(logeig)).mm(vh.transpose(1,0))
        return logX
    
    logA = log_mat(A)
    logB = log_mat(B)

    #logA = apply(A, torch.log)
    #logB = apply(B, torch.log)
    
    return euclid(logA, logB)

def euclid(A, B):
    
    diff = (A - B) 
    return (diff * diff).sum() / (4 * len(A)**2)

def affineinvariant(A, B):
    
    Binv = torch.inverse(B + 1e-10 * A.new_tensor(torch.eye(len(A))))
    return abs(A.mm(B)).mean()

def jeffrey(A, B):
    
    n = len(A)
    
    Ainv = torch.inverse(A)
    Binv = torch.inverse(B)
    
    return 0.5 * torch.trace(Ainv.mm(B)) + 0.5 * torch.trace(Binv.mm(A)) - n

def stein(A, B):
    
    N = len(A)

    reg = 1e-3 * A.new_tensor(torch.eye(N))
    
    arg_ApB = 0.5 * (A + B)  + reg
    arg_AB  = A.mm(B) 
    
    return stable_logdet(arg_ApB) - .5 * stable_logdet(arg_AB)

def riemann(A, B):
    
    B_nsqrt = apply(B, lambda x : x**(-.5))
    
    arg = B_nsqrt.mm(A).mm(B_nsqrt)
    
    return torch.log(arg - arg.min() + 1e-11).sum()

