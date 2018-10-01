import h5py
import torch
from torch import nn

import torch
import torch.nn.functional as F


class WalkerLoss(nn.Module):

    def forward(self, Psts, y):
        equality_matrix = torch.eq(y.clone().view(-1,1), y).float()
        p_target = equality_matrix / equality_matrix.sum(dim=1, keepdim=True)
        p_target.requires_grad = False

        L_walker = F.kl_div(torch.log(1e-8 + Psts), p_target, size_average=False)
        L_walker /= p_target.size()[0]

        return L_walker

class VisitLoss(nn.Module):

    def forward(self, Pt):
        p_visit = torch.ones([1, Pt.size()[1]]) / float(Pt.size()[1])
        p_visit.requires_grad = False
        if Pt.is_cuda: p_visit = p_visit.cuda()
        L_visit = F.kl_div(torch.log(1e-8 + Pt), p_visit, size_average=False)
        L_visit /= p_visit.size()[0]

        return L_visit

class AssociationMatrix(nn.Module):

    def __init__(self, verbose = False):
        super(AssociationMatrix, self).__init__()

        self.verbose = verbose

    def forward(self, xs, xt):
        """
        xs: (Ns, K, ...)
        xt: (Nt, K, ...)
        """

        # TODO not sure why clone is needed here
        Bs = xs.size()[0]
        Bt = xt.size()[0]

        xs = xs.clone().view(Bs, -1)
        xt = xt.clone().view(Bt, -1)

        W = torch.mm(xs, xt.transpose(1,0))

        # p(xt | xs) as softmax, normalize over xt axis
        Pst = F.softmax(W, dim=1) # Ns x Nt
        # p(xs | xt) as softmax, normalize over xs axis
        Pts = F.softmax(W.transpose(1,0), dim=1) # Nt x Ns

        # p(xs | xs)
        Psts = Pst.mm(Pts) # Ns x Ns

        # p(xt)
        Pt = torch.mean(Pst, dim=0, keepdim=True) # Nt

        return Psts, Pt

class AssociativeLoss(nn.Module):

    """ Association Loss for Domain Adaptation

    Reference:
    Associative Domain Adaptation, Hausser et al. (2017)
    """

    def __init__(self, walker_weight = 1., visit_weight = 1.):
        super(AssociativeLoss, self).__init__()

        self.matrix = AssociationMatrix()
        self.walker = WalkerLoss()
        self.visit  = VisitLoss()

        self.walker_weight = walker_weight
        self.visit_weight  = visit_weight

    def forward(self, xs, xt, y):

        Psts, Pt = self.matrix(xs, xt)
        L_walker = self.walker(Psts, y)
        L_visit  = self.visit(Pt)

        return self.visit_weight*L_visit + self.walker_weight*L_walker

class OTLoss(nn.Module):

    def __init__(self):

        super(OTLoss).__init__(self)

        self.mse_loss = nn.MSELoss()
        self.ce_loss  = nn.CrossEntropyLoss()

    def forward(self, xs, ys, xt, yt):

        self.mse_loss(xs, xt)
        self.ce_loss(ys, yt)

        pass

class WassersteinLoss(nn.Module):

    def __init__(self):

        super(WassersteinLoss).__init__(self)

        self.K = None

    def forward(self, input):
        pass

    
class Accuracy(nn.Module):

    def __init__(self):

        super(Accuracy).__init__(self)

    def forward(self, input):
        pass
    
    
class AugmentationLoss(nn.Module):
    """ Augmentation Loss from 
    https://github.com/Britefury/self-ensemble-visual-domain-adapt
    """
    
    def __init__(self, aug_loss_func = nn.MSELoss(), use_rampup=True):
        pass
    
    def forward(self):
        if self.use_rampup:
            unsup_mask = None
            conf_mask_count = None
            unsup_mask_count = None
        else:
            conf_tea = torch.max(tea_out, 1)[0]
            unsup_mask = conf_mask = torch.gt(conf_tea, confidence_thresh).float()
            unsup_mask_count = conf_mask_count = torch.sum(conf_mask)

        if loss == 'bce':
            aug_loss = network_architectures.robust_binary_crossentropy(stu_out, tea_out)
        else:
            d_aug_loss = stu_out - tea_out
            aug_loss = d_aug_loss * d_aug_loss

        aug_loss = torch.mean(aug_loss, 1)

        if self.use_rampup:
            unsup_loss = torch.mean(aug_loss) * rampup_weight_in_list[0]
        else:
            unsup_loss = torch.mean(aug_loss * unsup_mask)
                
        return unsup_loss
    
class ClassBalanceLoss(nn.Module):
    """ Class Balance Loss from 
    https://github.com/Britefury/self-ensemble-visual-domain-adapt
    """
    
    def forward(stu_out, tea_out):
        # Compute per-sample average predicated probability
        # Average over samples to get average class prediction
        avg_cls_prob = torch.mean(stu_out, 0)
        # Compute loss
        equalise_cls_loss = cls_bal_fn(avg_cls_prob, float(1.0 / n_classes))

        equalise_cls_loss = torch.mean(equalise_cls_loss) * n_classes

        if use_rampup:
            equalise_cls_loss = equalise_cls_loss * rampup_weight_in_list[0]
        else:
            if rampup == 0:
                equalise_cls_loss = equalise_cls_loss * torch.mean(unsup_mask, 0)

        return equalise_cls_loss * cls_balance
