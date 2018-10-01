import numpy as np
import torch
import torch.nn.functional as F
import math
from torch import nn

class RandomAffines():

    def __init__(self,
                 flip_x  = 0.5,
                 flip_y  = 0.5,
                 shear_x = (0,0.3),
                 shear_y = (0,0.3),
                 scale   = (0.8,1.4),
                 rotate  = (-math.pi/2, math.pi),
                 dx      = (-.2,.2),
                 dy      = (-.2,.2)
                ):

        self.__dict__.update(locals())

    def identify(self, size):
         return torch.Tensor([1, 0, 0, 0, 1, 0, 0, 0, 1]).view(-1,9).repeat(size,1).float()

    def reflect(self, size, p = .5):

        cx = torch.from_numpy(0 + (np.random.uniform(0,1,size = size) > self.flip_x)).float()
        cy = torch.from_numpy(0 + (np.random.uniform(0,1,size = size) > self.flip_y)).float()

        A = self.identify(size)
        A[:,0] = cx*2-1
        A[:,4] = cy*2-1

        return A

    def shear(self, size, p = .5):

        cx = torch.from_numpy(np.random.uniform(*self.shear_x,size = size)).float()
        cy = torch.from_numpy(np.random.uniform(*self.shear_y,size = size)).float()

        A = self.identify(size)
        A[:,1] = cx
        A[:,3] = cy

        return A

    def scaled(self, size, p = .5):

        cx = torch.from_numpy(np.random.uniform(*self.scale,size = size)).float()
        cy = torch.from_numpy(np.random.uniform(*self.scale,size = size)).float()

        A = self.identify(size)
        A[:,0] = cx
        A[:,4] = cy

        return A

    def rotated(self, size, p = .5):

        theta = torch.from_numpy(np.random.uniform(*self.rotate, size))

        A = self.identify(size)
        A[:,0] =  torch.cos(theta)
        A[:,1] =  torch.sin(theta)
        A[:,3] = -torch.sin(theta)
        A[:,4] =  torch.cos(theta)

        return A

    def shift(self, size, p = .5):

        dx    = torch.from_numpy(np.random.uniform(*self.dx, size))
        dy    = torch.from_numpy(np.random.uniform(*self.dy, size))

        A = self.identify(size)
        A[:,2] =  dx
        A[:,5] =  dy

        return A

    def matmul(self, A, B):

        A = A.view(-1,3,3)
        B = B.view(-1,3,3)


        return torch.bmm(A, B).view(-1, 9)

    def compose(self, size):

        order = [self.reflect,
                 self.rotated,
                 self.shear,
                 self.shift,
                 self.scaled,

                ]

        A = self.identify(size)
        for cmd in order:
            B = cmd(size)

            A = self.matmul(B, A)

        return A[:,:6]

class AffineTransformer(nn.Module):

    def __init__(self, *args, **kwargs):
        
        super(AffineTransformer,self).__init__()

        self.affines = RandomAffines(*args, **kwargs)

    def stn(self, x, theta):
        theta = theta.view(-1,2,3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def affine(self, y, theta):
        theta = theta.view(-1,2,3)

        B,C,_ = y.size()

        y = torch.cat([y,torch.ones(B,C,1)], dim=-1)
        y = torch.bmm(y, theta.transpose(2,1))

        return y

    def invert_affine(self, M):

        """
        Invert matrix for an affine transformation.
        Supports batch inputs

        M : Transformation matrices of shape (... x 6)

        Output: Inverse transformation matrices of shape (... x 6)
        """

        a,b,x,c,d,y = [M[...,i] for i in range(6)]

        D = a*d - b*c
        E = a*y - x*c

        a_ = 1/a * (1 + (c*b)/D)
        b_ = - b/D
        x_ = E/D * b/a - x/a
        c_ = -c / D
        d_ = a / D
        y_ = - E / D

        M_ = torch.stack([a_,b_,x_,c_,d_,y_], dim=-1)

        return M_.float()

    def __call__(self, x):

        n_squeeze = 0
        while x.dim() < 4:
            x = x.unsqueeze(0)
            n_squeeze += 1

        B,C,H,W = x.size()

        a = x.new_tensor([0,0])
        b = x.new_tensor([H,W])

        rescale = lambda x, a, b : (x - a)/(b - a) * 2 - 1
        scale   = lambda x, a, b : (x + 1)/2 * (b - a) + a

        Ay = x.new_tensor(self.affines.compose(B))

        ###

        Ax = self.invert_affine(Ay)
        x_ = self.stn(x, Ax)

        for _ in range(n_squeeze):
            x_ = x_.squeeze(0)
            y_ = y_.squeeze(0)

        return x_