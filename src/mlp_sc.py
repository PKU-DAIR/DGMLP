import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx

from utils import load_data, accuracy, normalize_adj
from models import ResMLP, DenseMLP


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    model.eval()
    output = model(features)

    acc_val = accuracy(output[idx_val], labels[idx_val])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print('Epoch: {:03d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'acc_test: {:.4f}'.format(acc_test.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return acc_val, acc_test


parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--hops', type=int, default=6)
parser.add_argument('--num_layers', type=int, default=6)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--dataset', type=str, default='pubmed')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test, _ = load_data(dataset=args.dataset)
n_nodes, feat_dim = features.shape
labels = torch.LongTensor(labels)

model = ResMLP(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
'''model = DenseMLP(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)'''

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
model = model.to(device)
features = features.to(device)
labels = labels.to(device)


t_total = time.time()
best_val = 0
best_test = 0
for epoch in range(args.epochs):
    acc_val, acc_test = train(epoch)
    if acc_val > best_val:
        best_val = acc_val
        best_test = acc_test

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
print(f'Best val: {best_val:.4f}, best test: {best_test:.4f}')
