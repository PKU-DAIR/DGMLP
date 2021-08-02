import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from layers import GraphConvolution


class ResMLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(ResMLP, self).__init__()
        self.fc1 = nn.Linear(nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nhid)
        self.fc3 = nn.Linear(nhid, nhid)
        self.fc4 = nn.Linear(nhid, nhid)
        self.fc5 = nn.Linear(nhid, nhid)
        self.fc6 = nn.Linear(nhid, nhid)
        self.fc7 = nn.Linear(nhid, nhid)
        self.fc8 = nn.Linear(nhid, nhid)
        self.fc9 = nn.Linear(nhid, nhid)
        self.fc10 = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x):
        x1 = F.dropout(x, self.dropout, training=self.training)
        x1 = F.relu(self.fc1(x1))
        x2 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.fc2(x2)) + x1
        x3 = F.dropout(x2, self.dropout, training=self.training)
        x3 = F.relu(self.fc2(x3)) + x2
        x4 = F.dropout(x3, self.dropout, training=self.training)
        x4 = F.relu(self.fc2(x4)) + x3
        x5 = F.dropout(x4, self.dropout, training=self.training)
        x5 = F.relu(self.fc2(x5)) + x4
        x6 = F.dropout(x5, self.dropout, training=self.training)
        x6 = F.relu(self.fc2(x6)) + x5
        x7 = F.dropout(x6, self.dropout, training=self.training)
        x7 = F.relu(self.fc2(x7)) + x6
        x8 = F.dropout(x7, self.dropout, training=self.training)
        x8 = F.relu(self.fc2(x8)) + x7
        x9 = F.dropout(x8, self.dropout, training=self.training)
        x9 = F.relu(self.fc2(x9)) + x8
        x10 = F.dropout(x9, self.dropout, training=self.training)
        x10 = self.fc10(x10)
        return F.log_softmax(x10, dim=1)


class DenseMLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(ResMLP, self).__init__()
        self.fc1 = nn.Linear(nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nhid)
        self.fc3 = nn.Linear(nhid, nhid)
        self.fc4 = nn.Linear(nhid, nhid)
        self.fc5 = nn.Linear(nhid, nhid)
        self.fc6 = nn.Linear(nhid, nhid)
        self.fc7 = nn.Linear(nhid, nhid)
        self.fc8 = nn.Linear(nhid, nhid)
        self.fc9 = nn.Linear(nhid, nhid)
        self.fc10 = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x):
        x1 = F.dropout(x, self.dropout, training=self.training)
        x1 = F.relu(self.fc1(x1))
        x2 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.fc2(x2)) + x1
        x3 = F.dropout(x2, self.dropout, training=self.training)
        x3 = F.relu(self.fc2(x3)) + x1 + x2
        x4 = F.dropout(x3, self.dropout, training=self.training)
        x4 = F.relu(self.fc2(x4)) + x1 + x2 + x3
        x5 = F.dropout(x4, self.dropout, training=self.training)
        x5 = F.relu(self.fc2(x5)) + x1 + x2 + x3 + x4
        x6 = F.dropout(x5, self.dropout, training=self.training)
        x6 = F.relu(self.fc2(x6)) + x1 + x2 + x3 + x4 + x5
        x7 = F.dropout(x6, self.dropout, training=self.training)
        x7 = F.relu(self.fc2(x7)) + x1 + x2 + x3 + x4 + x5 + x6
        x8 = F.dropout(x7, self.dropout, training=self.training)
        x8 = F.relu(self.fc2(x8)) + x1 + x2 + x3 + x4 + x5 + x6 + x7
        x9 = F.dropout(x8, self.dropout, training=self.training)
        x9 = F.relu(self.fc2(x9)) + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8
        x10 = F.dropout(x9, self.dropout, training=self.training)
        x10 = self.fc10(x10)
        return F.log_softmax(x10, dim=1)


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


class DenseGCN(nn.Module):
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
        x3 = F.relu(self.gc3(x3, adj)) + x1 + x2
        x4 = F.dropout(x3, self.dropout, training=self.training)
        x4 = F.relu(self.gc4(x4, adj)) + x1 + x2 + x3
        x5 = F.dropout(x4, self.dropout, training=self.training)
        x5 = F.relu(self.gc5(x5, adj)) + x1 + x2 + x3 + x4
        x6 = F.dropout(x5, self.dropout, training=self.training)
        x6 = F.relu(self.gc6(x6, adj)) + x1 + x2 + x3 + x4 + x5
        x7 = F.dropout(x6, self.dropout, training=self.training)
        x7 = F.relu(self.gc7(x7, adj)) + x1 + x2 + x3 + x4 + x5 + x6
        x8 = F.dropout(x7, self.dropout, training=self.training)
        x8 = F.relu(self.gc8(x8, adj)) + x1 + x2 + x3 + x4 + x5 + x6 + x7
        x9 = F.dropout(x8, self.dropout, training=self.training)
        x9 = F.relu(self.gc9(x9, adj)) + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8
        x10 = F.dropout(x9, self.dropout, training=self.training)
        x10 = F.relu(self.gc10(x10, adj))
        return F.log_softmax(x10, dim=1)


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        '''self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nhid)
        self.gc5 = GraphConvolution(nhid, nhid)
        self.gc6 = GraphConvolution(nhid, nhid)
        self.gc7 = GraphConvolution(nhid, nhid)
        self.gc8 = GraphConvolution(nhid, nhid)
        self.gc9 = GraphConvolution(nhid, nhid)'''
        self.gc10 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = F.relu(x)
        '''x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc4(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc5(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc6(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc7(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc8(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc9(x, adj)
        x = F.relu(x)'''
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc10(x, adj)
        return F.log_softmax(x, dim=1)
