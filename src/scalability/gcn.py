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
parser.add_argument('--hops', type=int, default=20,
                    help='number of hops')
parser.add_argument('--log_steps', type=int, default=1)
parser.add_argument('--device', type=int, default=3)
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

    model = GCN(feat_dim, args.hidden, 10, args.dropout).to(device)

    for run in range(args.runs):
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            train(model, input_features, y_true, adj_norm, optimizer)


if __name__ == '__main__':
    set_seed(args.seed)
    t1 = time.time()
    run(args)
    print(f'Total time: {round(time.time()-t1, 4)}s')
