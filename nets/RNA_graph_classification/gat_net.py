import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.gat_layer import GraphAttentionLayer, SpGraphAttentionLayer


class GAT(nn.Module):
    #def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
    def __init__(self, net_params):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        nfeat = net_params['nfeat']
        nhid = net_params['nhid']
        nclass = net_params['nclass']
        alpha = net_params['alpha']
        nheads = net_params['nheads']

        self.dropout = net_params['dropout']

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=net_params['dropout'], alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=net_params['dropout'], alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class SpGAT(nn.Module):
    def __init__(self, net_params):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        nfeat = net_params['nfeat']
        nhid = net_params['nhid']
        nclass = net_params['nclass']
        alpha = net_params['alpha']
        nheads = net_params['nheads']

        self.dropout = net_params['dropout']


        self.attentions = [SpGraphAttentionLayer(nfeat,
                                                 nhid,
                                                 dropout=net_params['dropout'],
                                                 alpha=alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads,
                                             nclass,
                                             dropout=net_params['dropout'],
                                             alpha=alpha,
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)