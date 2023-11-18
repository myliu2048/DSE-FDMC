import torch
import numpy as np
import torch.nn.functional as F


def infer(model, device, loader, args):
    '''
    Infer the commonZ and labels_vector
    '''
    model.eval()
    commonZ, labels_vector, P_pred = [], [], []
    for batch_idx, (xs, real_labels, _) in enumerate(loader):
        for v in range(args.view):
            xs[v] = xs[v].to(device) 
        with torch.no_grad():
            Hhat, _,_,_,_,_,_, p = model(xs)
            
        p_pred = p.cpu().detach().numpy().argmax(axis=1) 
        Hhat = Hhat.detach()
        commonZ.extend(Hhat.cpu().detach().numpy())
        labels_vector.extend(real_labels.numpy())
        P_pred.extend(p_pred)
    return np.array(commonZ), np.array(labels_vector), np.array(P_pred)
