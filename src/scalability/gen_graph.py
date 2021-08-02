import networkx as nx
import torch
import numpy as np
import scipy.sparse as sp

n_feat = 100
n_classes = 10
for n in [100000]:
    print(n)
    G = nx.fast_gnp_random_graph(n=n, p=0.0001)
    adj = nx.adjacency_matrix(G)
    features = torch.Tensor(n, n_feat)
    labels = torch.randint(n_classes, (1, n))
    torch.save(features, f"./data/graph_feat_{n}.pt")
    torch.save(labels, f'./data/graph_y_{n}.pt')
    torch.save(adj, f'./data/graph_adj_{n}.pt')
