"""
    Utility file to select GraphNN model as
    selected by the user
"""
from nets.RNA_graph_classification.gcn_net import GCNNet
from nets.RNA_graph_classification.graphsage_net import GraphSageNet
from nets.RNA_graph_classification.gat_net import GAT

def GCN(net_params):
    return GCNNet(net_params)

def GraphSage(net_params):
    return GraphSageNet(net_params)

def GAT(net_params):
    return GAT(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'GCN': GCN,
        'GraphSage': GraphSage,
        'GAT':GAT
    }
        
    return models[MODEL_NAME](net_params)