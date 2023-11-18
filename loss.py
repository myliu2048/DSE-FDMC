import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


class NMCL1(nn.Module):
    def __init__(self):
        super(NMCL1, self).__init__()

    def forward(self, args, z_i, z_j, S_pos): 
        if args.k == 1:
            args.k = 2 # args.k here must greater than 1
        S_neg = 1 - S_pos 
        adj2 = get_adj(z_j, args)
        nei_count = torch.sum(adj2, 1)  
        nei_count = torch.squeeze(nei_count).to('cuda')
        neg_mask = 1 - adj2 
        adj2 = adj2 - torch.diag_embed(adj2.diag())     
        intra_view_sim2 = torch.matmul(z_i, z_i.T) / args.temperature_Z
        pos = intra_view_sim2.mul(adj2)
        neg = torch.multiply(intra_view_sim2, S_neg).mul(neg_mask)
        loss =  pos.sum(1) / ( pos.sum(1) + neg.sum(1) + 1e-9)  / nei_count 
        return -torch.log(loss.mean())



class NMCL2(nn.Module):
    def __init__(self):
        super(NMCL2, self).__init__()

    def forward(self, args, h_i, h_j, P): 
        mask = (P == P.max(dim=1, keepdim=True)[0]).to(dtype=torch.int32)
        P = torch.mul(mask, P)
        adj = get_adj(P, args)   
        N = 2 * args.batch_size
        h = torch.cat((h_i, h_j), dim=0)
        sim = torch.matmul(h, h.T) / args.temperature_H
        sim_i_j = torch.diag(sim, args.batch_size)
        sim_j_i = torch.diag(sim, -args.batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1) 
        mask = ~(adj.repeat(2,2).bool())
        negative_samples = sim[mask].reshape(N, -1)
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = nn.CrossEntropyLoss(reduction="mean")(logits, labels)
        return loss
    

def get_adj(y, args):
    S = pairwise_distances(y.cpu().detach().numpy(), metric='euclidean')   
    S = S + 0.0001*np.random.random(size=[args.batch_size, args.batch_size]) - np.eye(args.batch_size)
    S = torch.from_numpy(-1*S)
    adj = (S >= torch.topk(S,args.k,dim=1)[0][:,-1].reshape(-1,1)).long() 
    return adj.to('cuda')

