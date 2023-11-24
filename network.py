from torch import nn
import torch.nn.functional as F
import torch


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024), 
            nn.BatchNorm1d(1024),
            nn.ReLU(True), 
            nn.Dropout(0.2),
            nn.Linear(1024, 1024), 
            nn.BatchNorm1d(1024),
            nn.ReLU(True), 
            nn.Dropout(0.2),
            nn.Linear(1024, 1024), 
            nn.BatchNorm1d(1024),
            nn.ReLU(True), 
            nn.Dropout(0.2),
            nn.Linear(1024, feature_dim), 
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 1024), 
            nn.ReLU(), 
            nn.Dropout(0.2),
            nn.Linear(1024, 1024), 
            nn.ReLU(), 
            nn.Dropout(0.2), 
            nn.Linear(1024, 1024), 
            nn.ReLU(), 
            nn.Dropout(0.2),
            nn.Linear(1024, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class Model(nn.Module):
    def __init__(self, args, device):
        view, input_size, low_feature_dim, high_feature_dim = \
            args.view, args.dims, args.low_feature_dim, args.high_feature_dim
        super(Model, self).__init__()
        self.encoders = []
        self.decoders = []
        self.Specific_view = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], low_feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], low_feature_dim).to(device))
            self.Specific_view.append(nn.Linear(low_feature_dim, high_feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.view = view
        self.TEL = nn.TransformerEncoderLayer(d_model=low_feature_dim * view, nhead=args.nhead,
                                                                  dim_feedforward=256)
        self.Common_view = nn.Sequential(
            nn.Linear(low_feature_dim * view, high_feature_dim)
        )

        self.clustering_layer = nn.Sequential(
            nn.Linear(high_feature_dim, args.class_num),
            nn.Softmax(dim=1)
        )

    def forward(self, xs):
        xrs, zs, hs = [], [], []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h = F.normalize(self.Specific_view[v](z), dim=1)
            xr = self.decoders[v](z)
            zs.append(z)
            xrs.append(xr)
            hs.append(h)     
        catZ = torch.cat(zs, 1)
        catZ = F.normalize(catZ, dim=1)
        fusedZ = catZ.unsqueeze(1)  
        fusedZ, S = self.TEL(fusedZ)  # built-in TransformerEncoderLayer doesn't output S by default.
        fusedZ = fusedZ.squeeze(1)  
        S = S.squeeze(0)  
        Hhat = F.normalize(self.Common_view(fusedZ), dim=1)
        p = self.clustering_layer(Hhat)
        return Hhat, S, xrs, zs, hs, catZ, fusedZ, p
