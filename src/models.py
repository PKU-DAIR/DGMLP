import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from layers import GraphConvolution


class Dense(nn.Module):
    def __init__(self, in_features, out_features, bias='bn'):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias == 'bn':
            self.bias = nn.BatchNorm1d(out_features)
        else:
            self.bias = lambda x: x
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
    def forward(self, input):
        output = torch.mm(input, self.weight)
        output = self.bias(output)
        if self.in_features == self.out_features:
            output = output + input
        return output


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return F.log_softmax(x, dim=-1)


class LR(nn.Module):
    def __init__(self, nfeat, nclass, dropout):
        super(LR, self).__init__()
        self.fc1 = nn.Linear(nfeat, nclass)
        self.dropout = dropout

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


class GCN_modified(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_modified, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj_1, adj_2):
        x = self.gc1(x, adj_1)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj_2)
        return F.log_softmax(x, dim=1)


class ResGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(ResGCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nhid)
        self.gc5 = GraphConvolution(nhid, nhid)
        self.gc6 = GraphConvolution(nhid, nhid)
        self.gc7 = GraphConvolution(nhid, nhid)
        self.gc8 = GraphConvolution(nhid, nhid)
        self.gc9 = GraphConvolution(nhid, nhid)
        self.gc10 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x1 = F.relu(self.gc1(x, adj))
        x2 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.gc2(x2, adj)) + x1
        x3 = F.dropout(x2, self.dropout, training=self.training)
        x3 = F.relu(self.gc3(x3, adj)) + x2
        x4 = F.dropout(x3, self.dropout, training=self.training)
        x4 = F.relu(self.gc4(x4, adj)) + x3
        x5 = F.dropout(x4, self.dropout, training=self.training)
        x5 = F.relu(self.gc5(x5, adj)) + x4
        x6 = F.dropout(x5, self.dropout, training=self.training)
        x6 = F.relu(self.gc6(x6, adj)) + x5
        x7 = F.dropout(x6, self.dropout, training=self.training)
        x7 = F.relu(self.gc7(x7, adj)) + x6
        x8 = F.dropout(x7, self.dropout, training=self.training)
        x8 = F.relu(self.gc8(x8, adj)) + x7
        x9 = F.dropout(x8, self.dropout, training=self.training)
        x9 = F.relu(self.gc9(x9, adj)) + x8
        x10 = F.dropout(x9, self.dropout, training=self.training)
        x10 = F.relu(self.gc10(x10, adj))
        return F.log_softmax(x10, dim=1)


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.bn3 = nn.BatchNorm1d(nhid)
        self.gc4 = GraphConvolution(nhid, nhid)
        self.bn4 = nn.BatchNorm1d(nhid)
        self.gc5 = GraphConvolution(nhid, nhid)
        self.bn5 = nn.BatchNorm1d(nhid)
        self.gc6 = GraphConvolution(nhid, nhid)
        self.bn6 = nn.BatchNorm1d(nhid)
        self.gc7 = GraphConvolution(nhid, nhid)
        self.bn7 = nn.BatchNorm1d(nhid)
        self.gc8 = GraphConvolution(nhid, nhid)
        self.bn8 = nn.BatchNorm1d(nhid)
        self.gc9 = GraphConvolution(nhid, nhid)
        self.bn9 = nn.BatchNorm1d(nhid)
        self.gc10 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        #x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc10(x, adj)
        return F.log_softmax(x, dim=1)
