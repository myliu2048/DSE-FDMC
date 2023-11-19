import torch
from network import Model
from sklearn.cluster import KMeans
from metric import evaluate
from infer import infer
import argparse
from dataloader import load_data
import torch.nn.functional as F
from config import config
import numpy as np
import random

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True 

datalist = ['Caltech-3V','Caltech-4V','Caltech-5V','MNIST_USPS','Prokaryotic','Hdigit']
data = datalist[3]
args = config(data)

setup_seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval(valid_loader, model, args):
    Hhat, gt_label,_ = infer(model, device, valid_loader, args)
    print('Clustering results:')
    kmeans = KMeans(n_clusters=args.class_num, n_init=100)
    y_pred = kmeans.fit_predict(Hhat)

    nmi, ari, acc, pur = evaluate(gt_label, y_pred)
    print('ACC = {:.4f} NMI = {:.4f}'.format(acc, nmi))

dataset, dims, view, data_size, class_num = load_data(args.dataset)
    
valid_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False)                                                       
 
model = Model(args, device)
model = model.to(device)
checkpoint = torch.load('./models/' + args.dataset + '.pth')
model.load_state_dict(checkpoint)
print("Dataset:{}".format(args.dataset))
print("Loading models...")

eval(valid_loader, model, args)



