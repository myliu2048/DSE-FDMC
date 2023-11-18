import argparse

def config(data='Caltech-3V'):
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('-data','--dataset', default=data)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('-temZ',"--temperature_Z", default=0.1, type=float)
    parser.add_argument('-temH',"--temperature_H", default=0.2, type=float)
    parser.add_argument("--learning_rate", default=0.0001, type=float)
    parser.add_argument("--workers", default=8)
    parser.add_argument("--epochs", default=50, type=int)  
    parser.add_argument("--mse_epochs", default=0, type=int)   
    parser.add_argument('-lfd',"--low_feature_dim", default=512, type=int)
    parser.add_argument('-hfd',"--high_feature_dim", default=128, type=int)
    parser.add_argument("--view", default=2)
    parser.add_argument("--lam1", default=1, type=float)
    parser.add_argument("--lam2", default=1, type=float)
    parser.add_argument("--k", default=3, type=int, help="Number of masked neighbors")  
    parser.add_argument("--nhead", default=1, type=int, 
                        help="Number of self-attention head, embed_dim must be divisible by num_heads")  

    args = parser.parse_args()

    if args.dataset == "MNIST_USPS":
        args.dims = [784, 784]
        args.class_num = 10
        args.seed = 1
        args.lam1 = 0.1
        args.temperature_H = 0.7
        args.mse_epochs = 50
        
    if args.dataset == 'Prokaryotic':
        args.dims = [438, 3, 393]
        args.class_num = 4
        args.seed = 1
        args.epochs = 30
        args.lam1 = 0.1
        args.view = 3
        
    if args.dataset == "Hdigit":
        args.dims = [784, 256]
        args.class_num = 10
        args.seed = 10
        args.view = 2
        args.temperature_H = 0.1  
        args.lam1 = 0.1
    # ======================================
    if args.dataset == "Caltech-5V":
        args.dims = [40, 254, 928, 512, 1984]
        args.view = 5
        args.class_num = 7
        args.seed = 10
        
    if args.dataset == "Caltech-4V":
        args.dims = [40, 254, 928, 512]
        args.view = 4
        args.class_num = 7
        args.seed = 10

    if args.dataset == "Caltech-3V":
        args.dims = [40, 254, 928]
        args.view = 3
        args.class_num = 7
        args.seed = 1
      
    return args