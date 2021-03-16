from torch_geometric.nn import GCNConv

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class AttentionModule(nn.Module):
    """
    SimGNN Attention Module to make a pass on graph.
    """

    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(AttentionModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = nn.Parameter(
            torch.Tensor(
                self.args.filters_3,
                self.args.filters_3,
            )
        )

    def init_parameters(self):
        """
        Initializing weights.
        """
        nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding):
        """
        Making a forward propagation pass to create a graph level representation.
        :param embedding: Result of the GCN  =>  [num_nodes, filters_3]
        :return representation: A graph level representation vector.
        """
        # global_context： 所有节点的embedding求平均
        # torch.matmul(embedding, self.weight_matrix) shape =  [num_nodes, filters_3] * [filters_3, filters_3] = [num_nodes, filters_3]
        # dim = 0 表示 列求平均
        # 所以 global_context 的shape 为 => [1, filters_3]  => [1, embedding_size] => [1, 32]
        # 也就是说， mean求得的是 所有节点的embedding表示的平均值
        global_context = torch.mean(torch.matmul(
            embedding, self.weight_matrix), dim=0)
        # transformed_global 就是 paper的 "C" , [1, embedding_size] => [1, 32]
        transformed_global = torch.tanh(global_context)
        # sigmoid_scores 的shape 为 [num_nodes, filters_3] * [filters_3, 1] = [num_nodes, 1]
        sigmoid_scores = torch.sigmoid(
            torch.mm(embedding, transformed_global.view(-1, 1)))

        # 为图的每一个节点都计算一个 att_embedding表示
        # representation.shape = [filters_3, num_nodes] * [num_nodes, 1] = [filters_3, 1]  =>  [32, 1]
        representation = torch.mm(torch.t(embedding), sigmoid_scores)
        return representation


class TenorNetworkModule(nn.Module):
    """
    SimGNN Tensor Network module to calculate similarity vector.
    """

    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(TenorNetworkModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        # [filters_3 , filters_3 , tensor_neurons]  =>  [32 , 32 , 16]
        self.weight_matrix = nn.Parameter(
            torch.Tensor(
                self.args.filters_3,
                self.args.filters_3,
                self.args.tensor_neurons,
            )
        )
        # [tensor_neurons , (2 * filters_3)]  =>  [16, 64]
        self.weight_matrix_block = nn.Parameter(torch.Tensor(self.args.tensor_neurons,
                                                             2*self.args.filters_3))
        #   [tensor_neurons , 1]  => [16 , 1]
        self.bias = nn.Parameter(
            torch.Tensor(self.args.tensor_neurons, 1))

    def init_parameters(self):
        """
        Initializing weights.
        """
        nn.init.xavier_uniform_(self.weight_matrix)
        nn.init.xavier_uniform_(self.weight_matrix_block)
        nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        :param embedding_1: Result of the 1st embedding after attention.
        :param embedding_2: Result of the 2nd embedding after attention.
        即： 传入的是两个图的 att_embedding  shape: [filters_3, 1]  =>  [32, 1]
        :return scores: A similarity score vector.
        """

        # calculate : scoring =  embedding_1 @ self.weight_matrix @ embedding_2
        # shape change:  T([32, 1]) * [32, 32*16]  =>  [1, 32] * [32, 32*16]  =>  [1, 32*16]  =>  [1, 512]
        scoring = torch.mm(torch.t(embedding_1),
                           self.weight_matrix.view(self.args.filters_3, -1))

        # [1, 512]  =>  [32, 16]
        scoring = scoring.view(self.args.filters_3, self.args.tensor_neurons)

        # [16, 32] * [32, 1]  =>  [16, 1]
        scoring = torch.mm(torch.t(scoring), embedding_2)

        # calculate: block_scoring =  embedding_1_2  @  self.weight_matrix_block
        # [64, 1]
        combined_representation = torch.cat((embedding_1, embedding_2))

        # [16, 64] * [64, 1]  =>  [16, 1]
        block_scoring = torch.mm(
            self.weight_matrix_block, combined_representation)

        # scoring + block_scoring
        # 对应值相加
        # [16, 1] + [16, 1] = [16, 1]
        scores = nn.functional.relu(scoring + block_scoring + self.bias)
        return scores


class SimGNN(pl.LightningModule):
    """
    SimGNN: A Neural Network Approach to Fast Graph Similarity Computation
    https://arxiv.org/abs/1808.05689
    """

    def __init__(self, args, number_of_labels):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super().__init__()
        self.args = args

        self.number_labels = number_of_labels
        self.setup_layers()

    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        if self.args.histogram == True:
            self.feature_count = self.args.tensor_neurons + self.args.bins
        else:
            # 16
            self.feature_count = self.args.tensor_neurons

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.calculate_bottleneck_features()
        # 三层GCN
        '''
        GCN init: def __init__(self, in_channels, out_channels)
        def forward(self, x, edge_index):
            x has shape [N, in_channels]
            edge_index has shape [2, E]
        '''
        self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1)
        self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
        self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)
        # 得到 [n, out_channels]

        # att
        self.attention = AttentionModule(self.args)

        # 用来计算 embedding_graph_1  和 embedding_graph_2 的合并向量
        self.tensor_network = TenorNetworkModule(self.args)

        # bottle-neck-neurons , 16
        # feature_count , 16
        # [16, 16]
        self.fully_connected_first = nn.Linear(self.feature_count,
                                               self.args.bottle_neck_neurons)
        # [16, 1]
        self.scoring_layer = nn.Linear(self.args.bottle_neck_neurons, 1)

    def calculate_histogram(self, abstract_features_1, abstract_features_2):
        """
        Calculate histogram from similarity matrix.
        :param abstract_features_1: Feature matrix for graph 1.
        :param abstract_features_2: Feature matrix for graph 2.
        :return hist: Histsogram of similarity scores.
        """
        scores = torch.mm(abstract_features_1, abstract_features_2).detach()
        scores = scores.view(-1, 1)
        hist = torch.histc(scores, bins=self.args.bins)
        hist = hist/torch.sum(hist)
        hist = hist.view(1, -1)
        return hist

    def convolutional_pass(self, edge_index, features):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Abstract feature matrix.
        """
        features = self.convolution_1(features, edge_index)
        features = nn.functional.relu(features)
        features = nn.functional.dropout(features,
                                         p=self.args.dropout,
                                         training=self.training)

        features = self.convolution_2(features, edge_index)
        features = nn.functional.relu(features)
        features = nn.functional.dropout(features,
                                         p=self.args.dropout,
                                         training=self.training)

        features = self.convolution_3(features, edge_index)
        return features

    def forward(self, data):
        """
        Forward pass with graphs.
        :param data: Data dictiyonary.
        :return score: Similarity score.
        """
        edge_index_1 = data["edge_index_1"]
        edge_index_2 = data["edge_index_2"]
        features_1 = data["features_1"]
        features_2 = data["features_2"]

        # 先使用图的邻接矩阵对特征矩阵进行GCN处理, 得到图节点的嵌入表示
        # [num_nodes, embedding_size]  =>  [num_nodes, filters_3]
        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)

        if self.args.histogram == True:
            hist = self.calculate_histogram(abstract_features_1,
                                            torch.t(abstract_features_2))

        # abstract_features_1 和 abstract_features_2是使用GCN对节点的嵌入表示后得到的矩阵
        # 所以注意力机制层是得到 att_embedding
        # 尺寸为 [num_nodes, GCN_out_channels]  即:  [num_nodes, filters_3]
        # pooled_features_1 和 pooled_features_2  =>  [filters_3, 1]
        pooled_features_1 = self.attention(abstract_features_1)
        pooled_features_2 = self.attention(abstract_features_2)

        # scores shape : [16, 1]
        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        # transpose: [1, 16]
        scores = torch.t(scores)

        if self.args.histogram == True:
            scores = torch.cat((scores, hist), dim=1).view(1, -1)

        # [1,16] @ [16, 16] = [1, 16]
        scores = nn.functional.relu(self.fully_connected_first(scores))
        # [1, 16] @ [16, 2] = [1, 2]
        # 返回每个类的概率
        return self.scoring_layer(scores)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

    def training_step(self, batch, _):
        # batches are processed one by one
        # so just get rid of the batch dimension
        batch = {k: v[0] for k, v in batch.items()}
        target = batch.pop('target').float()

        # compute loss
        prediction = torch.squeeze(self(batch))
        loss = F.binary_cross_entropy_with_logits(prediction, target)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, _):
        # batches are processed one by one
        # so just get rid of the batch dimension
        batch = {k: v[0] for k, v in batch.items()}
        target = batch.pop('target').float()

        # compute loss
        prediction = torch.squeeze(self(batch))
        loss = F.binary_cross_entropy_with_logits(prediction, target)

        self.log('val_loss', loss)
        return loss
