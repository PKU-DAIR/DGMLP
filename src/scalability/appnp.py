import argparse
import time

import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans, SpectralClustering
import networkx as nx
import math

from models import *
from utils import *


def prop(x, k, adj_norm):
    alpha = 0.5
    ori_x = x
    for _ in range(k):
        x = (1-alpha) * torch.spmm(adj_norm, x) + alpha * ori_x

    return x



class APPNP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, K,
                 dropout):
        super(APPNP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))
        self.k = K

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_norm):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        x = prop(x, self.k, adj_norm)
        return torch.log_softmax(x, dim=-1)


parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=100000)
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--hidden', type=int, default=256,
                    help='Number of units in hidden layer 1.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='value of weight decay')
parser.add_argument('--K', type=int, default=5,
                    help='number of hops')
parser.add_argument('--log_steps', type=int, default=1)
parser.add_argument('--device', type=int, default=4)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--runs', type=int, default=1,
                    help='number of run times')

args = parser.parse_args()


def train(model, x, y_true, adj_norm, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(x, adj_norm)
    loss = F.nll_loss(out, y_true)
    loss.backward()
    optimizer.step()

    return loss.item()


def run(args):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    features = torch.load(f'./data/graph_feat_{args.n}.pt')
    labels = torch.load(f'./data/graph_y_{args.n}.pt').squeeze(0)
    n_nodes, feat_dim = features.shape

    adj = torch.load(f'./data/graph_adj_{args.n}.pt')
    adj_norm = normalize_adj(adj, r=0.5)

    input_features = features.to(device)
    y_true = labels.to(device)
    adj_norm = adj_norm.to(device)

    model = APPNP(feat_dim, args.hidden, 10,
                args.num_layers, args.K, args.dropout).to(device)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            train(model, input_features, y_true, adj_norm, optimizer)


if __name__ == '__main__':
    t1 = time.time()
    set_seed(args.seed)
    run(args)
    print(f'Total time: {round(time.time()-t1, 4)}')
