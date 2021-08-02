import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy, normalize_adj
from models import GCN

# Training settings
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
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
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
adj, _ = normalize_adj(adj)
labels = torch.LongTensor(labels)

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)

gc1_list = []

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    for name, parms in model.named_parameters():
        if name == 'gc1.weight':
            gc1 = parms.grad.abs()
            gc1_list.append(round(gc1.cpu().data.mean().item(), 8))
    optimizer.step()

    model.eval()
    output = model(features, adj)

    acc_val = accuracy(output[idx_val], labels[idx_val])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print('Epoch: {:03d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'acc_test: {:.4f}'.format(acc_test.item()),
          'time: {:.4f}s'.format(time.time() - t))
    #return acc_val, acc_test, gc1, gc2, gc3, gc4, gc5, gc6, gc7
    return acc_val, acc_test


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
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
print(gc1_list)
