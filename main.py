import torch
from torch import nn
from network import Model
import numpy as np
import os, time
import random
from loss import NMCL1, NMCL2
from dataloader import load_data
from utils import logsetting
from tabulate import tabulate
from config import config
from infer import infer
from sklearn.cluster import KMeans
from metric import evaluate
from scipy.stats import entropy  

def pretrain(train_loader, model, criterion, optimizer, epoch, args, log):
    tot_loss = 0.
    model.train()  
    time0 = time.time()
    for batch_idx, (xs, real_labels, _) in enumerate(train_loader):
        for v in range(args.view):
            xs[v] = xs[v].to(device)
        labels = real_labels.to(device)
        optimizer.zero_grad()           
        loss_list = []
        Hhat, S, xrs, zs, hs, catZ, fusedZ, y = model(xs)
        # reconstruction loss
        for v in range(args.view):
            loss_list.append(criterion['MSE'](xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    epoch_time = time.time() - time0
    if epoch == 1 or epoch % 10 == 0:
        log.info("=======> PreTraining epoch: {}/{}, Loss:{:.6f}".format(epoch, args.mse_epochs,tot_loss/len(train_loader)))
    return epoch_time


def train(train_loader, model, criterion, optimizer, epoch, args, log):
    tot_loss = 0.
    model.train()  
    time0 = time.time()
    for batch_idx, (xs, real_labels, _) in enumerate(train_loader):
        for v in range(args.view):
            xs[v] = xs[v].to(device)
        labels = real_labels.to(device)
        optimizer.zero_grad()           
        loss_list = []
        Hhat, S, xrs, zs, hs, catZ, fusedZ, p = model(xs)
        # reconstruction loss
        for v in range(args.view):
            loss_list.append(criterion['MSE'](xs[v], xrs[v]))
        # low level contrastive loss
        loss_list.append(args.lam1*criterion['NMCL1'](args, catZ, fusedZ, S))

        kl = []
        for v in range(args.view):
            qi =  model.pseudo_clustering_layer(hs[v])
            kl.append(entropy(p.cpu().detach().numpy(),
                              qi.cpu().detach().numpy()).sum())
        w = kl / np.sum(kl)
        
        # high level contrastive loss
        for v in range(args.view):
            loss_list.append(args.lam2*w[v]*criterion['NMCL2'](args, hs[v], Hhat, p))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    epoch_time = time.time() - time0
    if epoch == 1 or epoch % 10 == 0:
        log.info("=======> Training epoch: {}/{}, Loss:{:.6f}".format(epoch, args.epochs,tot_loss/len(train_loader)))
    return epoch_time

def eval(valid_loader, model, args, log):
    Hhat, gt_label, _ = infer(model, device, valid_loader, args)
    kmeans = KMeans(n_clusters=args.class_num, n_init=100)
    y_pred = kmeans.fit_predict(Hhat)
    nmi, ari, acc, pur = evaluate(gt_label, y_pred)
    log.info('Clustering results: ACC = {:.4f} NMI = {:.4f} PUR={:.4f}'.format(acc, nmi, pur))


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True  


def main():
    setup_seed(args.seed)
    dataset, dims, view, data_size, class_num = load_data(args.dataset)
    train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True)
    valid_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False)
    if not os.path.exists('./models'):
        os.makedirs('./models')
    model = Model(args, device)
    model = model.to(device)

    criterion = {}
    criterion['MSE'] = nn.MSELoss().to(device)
    criterion['NMCL1'] = NMCL1().to(device)
    criterion['NMCL2'] = NMCL2().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate) 
    log = logsetting(args)
    log.info('**************** Time = {} ****************'
        .format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
    hparams_head = ['Hyper-parameters', 'Value']
    log.info(tabulate(vars(args).items(), headers=hparams_head))

    log.info("******** Training begin ********")
    train_time = 0
    epoch = 1
    while epoch <= args.mse_epochs:     
        epoch_time = pretrain(train_loader, model, criterion, optimizer, epoch, args, log)
        train_time += epoch_time
        epoch += 1
    
    epoch = 1
    while epoch <= args.epochs:     
        epoch_time = train(train_loader, model, criterion, optimizer, epoch, args, log)
        train_time += epoch_time
        epoch += 1
    eval(valid_loader, model, args, log)
    log.info('******** Training End, training time = {} s ********'.format(round(train_time, 2)))
    state = model.state_dict()
    torch.save(state, './models/' + args.dataset + '.pth')
    print('Saving model...') 

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datalist = ['Caltech-3V','Caltech-4V','Caltech-5V','MNIST_USPS','Prokaryotic','Hdigit']
    data = datalist[1]
    args = config(data)
    main()
