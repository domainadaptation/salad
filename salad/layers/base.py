import torch 
from torch import nn

class KLDivWithLogits(nn.Module):

    def __init__(self):

        super(KLDivWithLogits, self).__init__()

        self.kl = nn.KLDivLoss(size_average=False, reduce=True)
        self.logsoftmax = nn.LogSoftmax(dim = 1)
        self.softmax = nn.Softmax(dim = 1)


    def forward(self, x, y):

        log_p = self.logsoftmax(x)
        q     = self.softmax(y)

        return self.kl(log_p, q) / x.size()[0]


class AccuracyScore(nn.Module):
    
    def forward(self, y, t):
        
        with torch.no_grad():
            idc = y.detach().max(dim = -1)[1]
            acc = torch.eq(idc, t).float()
            acc = acc.mean()
        
        return acc 
    
class MeanAccuracyScore(nn.Module):
    
    def forward(self, y, t):

        with torch.no_grad():
            t = t.cpu()
            labels = torch.unique(t)
            
            idc = y.detach().max(dim = -1)[1].cpu()
            acc = torch.eq(idc, t).float()
            
            mean_acc = sum(acc[torch.eq(t,l)].mean() for l in labels) / len(labels)
            
        return mean_acc

class WeightedCE(nn.Module):
    """ Adapted from Self-Ensembling repository
    """
    
    def __init__(self, confidence_threshold = 0.96837722):
        
        super().__init__()
        
        self.threshold  =  0.96837722
        # NOTE changed to dim=1 from dim=-1
        self.softmax    = nn.Softmax(dim=1)
        
    def robust_binary_crossentropy(self, pred, tgt):
        inv_tgt = -tgt + 1.0
        inv_pred = -pred + 1.0 + 1e-6
        return -(tgt * torch.log(pred + 1.0e-6) + inv_tgt * torch.log(inv_pred))
        
    def forward(self, logits, logits_target):
        
        p_         = self.softmax(logits)
        p          = self.softmax(logits_target)
        conf, _    = p.max(dim=1)
        mask       = torch.gt(conf, self.threshold)
        
        if mask.sum() > 0:
            loss = self.robust_binary_crossentropy(p_[mask], p[mask]).mean()
            return loss
        
        # TODO this is a hack. replace by sth better
        loss = self.robust_binary_crossentropy(p_[:1], p[:1]).mean()
        return 0 * loss