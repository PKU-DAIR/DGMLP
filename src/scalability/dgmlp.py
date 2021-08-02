import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx

from utils import load_data, accuracy, normalize_adj
from models import LR


def train(epoch):
    model.train()

    optimizer.zero_grad()
    output = model(features)
    loss = F.nll_loss(output, labels)
    loss.backward()
    optimizer.step()

    return loss.item()


parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=100000)
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.2,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-3,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--hops', type=int, default=20)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--device', type=int, default=5)
parser.add_argument('--dataset', type=str, default='cora')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

t1 = time.time()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
features = torch.load(f'./data/graph_feat_{args.n}.pt')
labels = torch.load(f'./data/graph_y_{args.n}.pt').squeeze(0)
n_nodes, feat_dim = features.shape

adj = torch.load(f'./data/graph_adj_{args.n}.pt')
adj_norm = normalize_adj(adj)

G = nx.from_scipy_sparse_matrix(adj)
for i in range(n_nodes):
    if i not in G.nodes():
        G.add_node(i)
S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
print(len(S))
fea_inf = [torch.zeros(feat_dim)]*n_nodes
for g in S:
    adj_temp = nx.to_scipy_sparse_matrix(g)
    node_sum = adj_temp.shape[0]
    edge_sum = adj_temp.sum()/2
    row_sum = (adj_temp.sum(1) + 1)
    temp = row_sum/ (2*edge_sum+node_sum)
    temp = torch.Tensor(temp)
    res = torch.zeros(n_nodes)
    for node, tp in zip(g.nodes(), temp):
        res[node] = tp
    res = res.view(-1, n_nodes)
    res = torch.mm(res, features)
    for node in g.nodes():
        fea_inf[node] = res
fea_inf = torch.cat(fea_inf, dim=0)

feature_list = []
feature_list.append(features)
for _ in range(args.hops):
    feature_list.append(torch.spmm(adj_norm, feature_list[-1]))

weight_list_1 = []
weight_list_2 = []
norm_ori = torch.norm(features, 2, 1).add(1e-10)
norm_inf = torch.norm(fea_inf, 2, 1).add(1e-10)
for fea in feature_list:
    norm_cur = torch.norm(fea, 2, 1).add(1e-10)

    temp = torch.div((features*fea).sum(1), norm_ori)
    temp  = torch.div(temp, norm_cur)
    weight_list_1.append(temp.unsqueeze(-1))

    temp = torch.div((fea_inf*fea).sum(1), norm_inf)
    temp  = torch.div(temp, norm_cur)
    weight_list_2.append(temp.unsqueeze(-1))

alpha = torch.cat(weight_list_1, dim=1)
beta = torch.cat(weight_list_2, dim=1)

weight = F.softmax(alpha*(1-beta), dim=1)

input_features = []
for i in range(n_nodes):
    fea = 0.
    for j in range(args.hops+1):
        fea += (weight[i][j]*feature_list[j][i]).unsqueeze(0)
    input_features.append(fea)
input_features = torch.cat(input_features, dim=0)

model = LR(nfeat=features.shape[1],
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
model = model.to(device)
features = input_features.to(device)
labels = labels.to(device)

best_val = 0
best_test = 0
for epoch in range(args.epochs):
    train(epoch)

print("Total time: {:.4f}s".format(time.time() - t1))
