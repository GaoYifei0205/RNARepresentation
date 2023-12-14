import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np

"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    http://arxiv.org/abs/1609.02907
"""
from layers.gcn_layer import GCNLayer, ConvReadoutLayer, GNNPoolLayer, WeightCrossLayer
from layers.mlp_readout_layer import MLPReadout
from layers.conv_layer import ConvLayer, MAXPoolLayer
from layers.gat_layer import GraphAttentionLayer, CustomGATLayer, CustomGATLayerEdgeReprFeat, CustomGATLayerIsotropic


class GCNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        num_heads = net_params['n_heads']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        self.device = net_params['device']
        self.n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.n_classes = net_params['n_classes']
        # self.pre_gnn, self.pre_cnn = None, None
        self.pre_gnn = None
        self.base_weight = None
        self.node_weight = None
        self.sequence = None
        self.filter_out = None

        self.layer_type = {
            "dgl": GraphAttentionLayer,
            "edgereprfeat": CustomGATLayerEdgeReprFeat,
            "edgefeat": CustomGATLayer,
            "isotropic": CustomGATLayerIsotropic,
        }.get(net_params['layer_type'], GraphAttentionLayer)

        window_size = 501
        conv_kernel1, conv_kernel2 = [9, 4], [9, 1]
        conv_padding, conv_stride = [conv_kernel1[0] // 2, 0], 1
        pooling_kernel = [3, 1]
        pooling_padding, pooling_stride = [pooling_kernel[0] // 2, 0], 2
        # ceil 对浮点数向上取整
        width_o1 = math.ceil((window_size - conv_kernel1[0] + 2 * conv_padding[0] + 1) / conv_stride)
        width_o1 = math.ceil((width_o1 - pooling_kernel[0] + 2 * pooling_padding[0] + 1) / pooling_stride)
        width_o2 = math.ceil((width_o1 - conv_kernel2[0] + 2 * conv_padding[0] + 1) / conv_stride)
        width_o2 = math.ceil((width_o2 - pooling_kernel[0] + 2 * pooling_padding[0] + 1) / pooling_stride)

        # GNN start
        self.embedding_h = nn.Linear(in_dim, hidden_dim)  # in_dim由输入张量的形状决定，out_dim决定了输出张量的形状
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        # print("n_layers is ", self.n_layers) n_layers = 2

        self.layers_gnn = nn.ModuleList()
        # self.layers_gnn = nn.ModuleList([self.layer_type(hidden_dim * num_heads, hidden_dim, num_heads,
        #                                              dropout, self.batch_norm, self.residual) for _ in
        #                                             range(self.n_layers * 2 - 1)])
        # self.layers_gnn.append(self.layer_type(hidden_dim * num_heads, out_dim, 1, dropout, self.batch_norm))

        self.layers_gnn.append(GCNLayer(hidden_dim, hidden_dim, F.leaky_relu, dropout, self.batch_norm))
        # for _ in range(self.n_layers * 2 - 2):
        #     # self.layers_gnn.append(self.layer_type(hidden_dim * num_heads, hidden_dim, num_heads, dropout, self.batch_norm))
        #     self.layers_gnn.append(
        #         GCNLayer(hidden_dim, hidden_dim, F.leaky_relu, dropout, self.batch_norm, self.residual))
        # self.layers_gnn.append(GCNLayer(hidden_dim, hidden_dim, F.leaky_relu, dropout, self.batch_norm, self.residual))

        # self.layers_gnn.append(GCNLayer(hidden_dim, out_dim, F.leaky_relu, dropout, self.batch_norm, self.residual))

        # self.layers_gnn.append(GNNPoolLayer())
        # self.layers_gnn.append(GCNLayer(hidden_dim, hidden_dim, F.leaky_relu, dropout, self.batch_norm, self.residual))
        # self.layers_gnn.append(GNNPoolLayer())
        # GNN end

        # # CNN start
        self.conv_readout_layer = ConvReadoutLayer(self.readout)
        # self.layers_cnn = nn.ModuleList()
        # self.layers_cnn.append(
        #     ConvLayer(1, 32, conv_kernel1, F.leaky_relu, self.batch_norm, residual=False, padding=conv_padding))
        # for _ in range(self.n_layers - 1):
        #     self.layers_cnn.append(
        #         ConvLayer(32, 32, conv_kernel2, F.leaky_relu, self.batch_norm, residual=False, padding=conv_padding))
        #
        # self.layers_pool = nn.ModuleList()
        # for _ in range(self.n_layers):
        #     self.layers_pool.append(MAXPoolLayer(pooling_kernel, stride=pooling_stride, padding=pooling_padding))
        # # CNN end

        # self.cross_weight_layer = nn.ModuleList()
        # self.cross_weight_layer.append(WeightCrossLayer(in_dim=501, out_dim=501//2+1))
        # self.cross_weight_layer.append(WeightCrossLayer(in_dim=501, out_dim=501//4+1))
        self.batchnorm_weight = nn.BatchNorm1d(501)

        input_dim = width_o2 * 32
        # input_dim = 2016
        # self.MLP_layer = MLPReadout(501*32 + input_dim, self.n_classes)
        # self.MLP_layer = MLPReadout(501*32, self.n_classes)
        self.MLP_layer = MLPReadout(32, self.n_classes)
        # 501*32 + 2016 = 18048

    def forward(self, g, h, id, e):  # g:batch_graphs, h: batch_x节点特征, e: batch_e边特征
        # 详见train_RNAGraph_graph_classification.py
        batch_size = len(g.batch_num_nodes())  #128
        window_size = g.batch_num_nodes()[0] #tensor(501, device='cuda0')
        similar_loss = 0
        cnn_node_weight = 0
        weight2gnn_list = []
        weight2cnn_list = []

        # h2 = self._graph2feature(g)
        # self.sequence = h2
        # h2 = h2.to(self.device)
        # h2: torch.Size([128, 1, 501, 25])

        h1 = self.embedding_h(h)  # h1:(501*batch_size, 32)  h:(501*batch_size, 768)
        # print(h1.shape)
        h1 = self.in_feat_dropout(h1)  # h1:(501*batch_size, 32)
        # print(h1.shape)
        # print("loop start")
        # h2 = torch.unsqueeze(feature, dim=1)
        for i in range(self.n_layers):
            # GNN
            h1 = self.layers_gnn[i](g, h1)
            # h1 = self.layers_gnn[2 * i + 1](g, h1)
            # g, h1, _ = GNNPoolLayer(batch_size=batch_size, node_num=math.ceil(window_size / 2 ** i))(g, h1)

            # CNN
            # h2 = self.layers_cnn[i](h2)  # torch.Size([128, 32, 501, 22])
            # if i == 0:
            #     self.filter_out = h2
            #     cnn_node_weight = torch.mean(h2, dim=1).squeeze(-1)
            #     self.base_weight = self.batchnorm_weight(cnn_node_weight)
            #     cnn_node_weight = torch.sigmoid(self.batchnorm_weight(cnn_node_weight))  # torch.Size([128, 501, 22])
            #     # cnn_node_weight = cnn_node_weight.detach()
            #
            # h2 = self.layers_pool[i](h2)  # torch.Size([128, 32, 251, 11])

            # weight cross
            # print("shape", h2.shape)
            # weight2gnn = torch.flatten(h2.squeeze(-1).permute(0, 2, 1), end_dim=1)
            # weight2gnn_list.append(torch.mean(weight2gnn, dim=1).unsqueeze(-1))

            # weight2cnn = torch.mean(self.conv_readout_layer(g, h1), dim=1).squeeze(-1)
            # weight2cnn = self.batchnorm_weight(weight2cnn)
            # weight2cnn_list.append(weight2cnn)

        #     weight2cnn = self.cross_weight_layer[i](weight2cnn_list[-1].squeeze(-1))
        #     h2 = torch.mul(h2, weight2cnn.unsqueeze(1).unsqueeze(-1))
        #
        # similar_loss += torch.mean(torch.norm(cnn_node_weight - weight2cnn_list[-1], dim=1))
        # self.similar_loss = similar_loss

        g.ndata['h'] = h1  #更新节点特征
        g.ndata['id'] = id
        hg = dgl.mean_nodes(g,'h')
        # hg = self.conv_readout_layer(g, h1)  #(128,32,501,1)

        # gnn_node_weight = torch.mean(hg, dim=1).squeeze(-1) #(128,501)
        # self.node_weight = self.batchnorm_weight(gnn_node_weight) #(128,501)
        # hg.shape: torch.Size([128, 32, 501, 1])
        # cnn_node_weight.shape: torch.Size([128, 501, 22])
        # cnn_node_weight.unsqueeze(1).unsqueeze(-1).shape: torch.Size([128, 1, 501, 22, 1])
        # w = torch.sum(cnn_node_weight.unsqueeze(1), dim=3).unsqueeze(-1)
        # hg = torch.mul(hg, w)
        # hg = torch.mul(hg, cnn_node_weight.unsqueeze(1).unsqueeze(-1))

        # hg = torch.flatten(hg, start_dim=1)  # (128,501*32)
        # hc = torch.flatten(h2, start_dim=1)

        # h_final = torch.cat([hg, hc], dim=1)
        h_final = hg  # (128,501*32) 改后为#(128, 32)
        pred = self.MLP_layer(h_final) #(128,2)

        return pred

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label) #pred:(batch_size,2) label:(batch_size)
        # loss = criterion(self.pre_cnn, label) + criterion(self.pre_gnn, label)
        # loss += 0.01 * self.similar_loss
        return loss

    def _graph2feature(self, g):
        feat = g.ndata['h']
        start, first_flag = 0, 0
        for batch_num in g.batch_num_nodes():
            if first_flag == 0:
                output = torch.mean(torch.transpose(feat[start:start + batch_num], 1, 0).unsqueeze(0), dim = 2)
                first_flag = 1
            else:
                gr_mean = torch.mean(torch.transpose(feat[start:start + batch_num], 1, 0).unsqueeze(0), dim = 2)
                output = torch.cat([output,gr_mean], dim=0)
            start += batch_num
        # output = torch.transpose(output, 1, 2)
        # output = output.unsqueeze(1)
        return output
    
    def cosine_similarity(self, x, y):
        num = x.dot(y.T)
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        return num / denom
    
    def RBP_loss(self, g, label):
        result=["ALKBH5_Baltz2012",
            "C17ORF85_Baltz2012",
            "C22ORF28_Baltz2012",
            "CAPRIN1_Baltz2012",
            "CLIPSEQ_AGO2",
            "CLIPSEQ_ELAVL1",
            "CLIPSEQ_SFRS1",
            "ICLIP_HNRNPC",
            "ICLIP_TDP43",
            "ICLIP_TIA1",
            "ICLIP_TIAL1",
            "PARCLIP_AGO1234",
            "PARCLIP_ELAVL1",
            "PARCLIP_ELAVL1A",
            "PARCLIP_EWSR1",
            "PARCLIP_FUS",
            "PARCLIP_HUR",
            "PARCLIP_IGF2BP123",
            "PARCLIP_MOV10_Sievers",
            "PARCLIP_PUM2",
            "PARCLIP_QKI",
            "PARCLIP_TAF15",
            "PTBv1",
            "ZC3H7B_Baltz2012"]
        start, first_flag = 0, 0
        feat = g.ndata['feat']
        ids = g.ndata['id']
        w = self.embedding_h.weight #(32,768)
        b = self.embedding_h.bias #(32)
        w = w.to(self.device)
        b = b.to(self.device)
        for batch_num in g.batch_num_nodes():
            
            id = ids[start:start + batch_num]
            id_tensor = id[0]
            indices = torch.nonzero(id_tensor == 1).squeeze()
            index_as_int = indices.item()
            name = result[index_as_int]
            protein_feature = torch.load('/amax/data/gaoyifei/GraphProt/GraphProt_CLIP_sequences/'+ name +'/'+ name +'.pt')['representations'][12]
            mean_protein = torch.mean(protein_feature,dim = 0, keepdim=True).to(self.device)
            mean_protein_linear = torch.matmul(mean_protein,w.T)+b
            if first_flag == 0:
                protein_features = mean_protein_linear
                first_flag = 1
            else:
                protein_features = torch.cat([protein_features,mean_protein_linear], dim=0)
            start += batch_num
            
        criterion = nn.CosineEmbeddingLoss(margin = 0.1)

        graph_feature = self._graph2feature(g).to(self.device)

        cos_sim = self.cosine_similarity(graph_feature.cpu().detach().numpy(), protein_features.cpu().detach().numpy())
        label2 = torch.where(label == 0, -1, label) #原来标签为0和1，转为-1和1
        rbp_loss = criterion(graph_feature, protein_features.to(self.device), label2.to(self.device))

        return rbp_loss,cos_sim

# self.similar_loss = torch.norm(
#     torch.mean(torch.mean(h2, dim=1).squeeze(-1), dim=0) -
#     torch.mean(weight2cnn_list[-1].squeeze(1).squeeze(-1), dim=0))
