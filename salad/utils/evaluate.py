import numpy as np
import tqdm
import torch

def _predictions(model, data_loader, domain):
    labels = []
    preds  = []
    feats  = []

    for x,y in tqdm.tqdm(data_loader):
        x = x.cuda()
        y = y.cuda()

        if domain is not None:
            f,p = model(x, domain)
        else: 
            f,p = model(x)

        labels.append(y.cpu().numpy())
        preds.append(p.detach().cpu().numpy())
        feats.append(f.detach().cpu().numpy())

    labels = np.concatenate(labels, axis=0)
    preds = np.concatenate(preds, axis=0)
    feats = np.concatenate(feats, axis=0)
    
    return labels, preds,feats

def evaluate(checkpoints, data_loader, domain):

    labels, preds, feats = [], [], []
    
    for i, chk in enumerate(checkpoints):
        
        model = torch.load(chk, map_location=lambda storage, loc: storage)

        model.cuda()
        model.eval()

        l,p,f = _predictions(model, data_loader, domain)
        labels.append(l)
        preds.append(p)
        feats.append(f)
        
    labels = np.stack(labels, axis=0)
    preds = np.stack(preds, axis=0)
    feats = np.stack(feats, axis=0)
    
    return labels, preds, feats