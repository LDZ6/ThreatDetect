import hashlib
import json
import os

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import tree_sitter
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from tree_sitter import Language

# 指定语言并初始化解析器
LANGUAGES = {
    'php': Language('Webshell_detect_models/GCN/build/my-languages.so', 'php'),
    'java': Language('Webshell_detect_models/GCN/build/my-languages.so', 'java'),
}


def parse_code_to_graph(file_path, language_parser):
    G = nx.DiGraph()

    with open(file_path, 'rb') as f:
        content = f.read()

    try:
        # 直接使用字节对象，无需解码
        tree = language_parser.parse(content)

        for node in tree.root_node.children:
            G.add_node(node.type)

            for child_node in node.children:
                G.add_edge(node.type, child_node.type)

    except Exception as e:
        print(f"Error in file: {file_path} - {e}")

    return G


def save_graph_as_json(graph, output_folder, file_name):
    file_hash = hashlib.md5(json.dumps(nx.to_dict_of_lists(graph)).encode()).hexdigest()
    json_file_path = os.path.join(output_folder, f'{file_hash}_Graph.json')

    with open(json_file_path, 'w') as json_file:
        data = nx.readwrite.json_graph.node_link_data(graph)
        json.dump(data, json_file, indent=2)


def extract_graph_features(file_path):
    # 根据文件扩展名选择语言
    file_sec, file_extension = os.path.splitext(file_path)
    language = None

    if file_extension == '.php':
        language = 'php'
    elif file_extension == '.jsp':
        language = 'java'

    if language:
        language_parser = tree_sitter.Parser()
        language_parser.set_language(LANGUAGES[language])

        graph = parse_code_to_graph(file_path, language_parser)
        data = nx.readwrite.json_graph.node_link_data(graph)
        return data


def detect_process_single_graph(data_path, w2v_model):
    graph_data = extract_graph_features(data_path)

    # 将节点信息转换为矢量表示
    nodes = [node['id'] for node in graph_data['nodes']]
    node_index_map = {node: index for index, node in enumerate(nodes)}
    node_vectors = [w2v_model.wv[node['id']] if node['id'] in w2v_model.wv else np.zeros(w2v_model.vector_size) for node
                    in graph_data['nodes']]
    node_vectors = np.array(node_vectors)

    # 构建边缘索引
    edge_index = [[node_index_map[node['source']], node_index_map[node['target']]] for node in graph_data['links']]
    edge_index = np.array(edge_index).T

    # 标签：1 表示 webshell，0 表示正常
    label = 1 if 'webshell' in data_path else 0

    # 构建 PyTorch Geometric Data 对象
    data = Data(x=torch.tensor(node_vectors, dtype=torch.float32),
                edge_index=torch.tensor(edge_index, dtype=torch.long).contiguous(),
                num_nodes=len(nodes),
                y=torch.tensor([label], dtype=torch.float32))

    return data


def process_single_graph(data_path, w2v_model):
    with open(data_path, 'r') as f:
        graph_data = json.load(f)

    # 将节点信息转换为矢量表示
    nodes = [node['id'] for node in graph_data['nodes']]
    node_index_map = {node: index for index, node in enumerate(nodes)}
    node_vectors = [w2v_model.wv[node['id']] if node['id'] in w2v_model.wv else np.zeros(w2v_model.vector_size) for node
                    in graph_data['nodes']]
    node_vectors = np.array(node_vectors)

    # 构建边缘索引
    edge_index = [[node_index_map[node['source']], node_index_map[node['target']]] for node in graph_data['links']]
    edge_index = np.array(edge_index).T

    # 标签：1 表示 webshell，0 表示正常
    label = 1 if 'webshell' in data_path else 0

    # 构建 PyTorch Geometric Data 对象
    data = Data(x=torch.tensor(node_vectors, dtype=torch.float32),
                edge_index=torch.tensor(edge_index, dtype=torch.long).contiguous(),
                num_nodes=len(nodes),
                y=torch.tensor([label], dtype=torch.float32))

    return data


# 定义 GCN 分类器
class GCNClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, dropout_rate=0.5):
        super(GCNClassifier, self).__init__()

        # GCN 层
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, output_dim)

        # 图形池化层
        self.pooling = global_mean_pool

        # LeakyReLU 激活函数
        self.leaky_relu = nn.LeakyReLU()

        # 全连接层
        self.fc1 = nn.Linear(output_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # GCN 层
        x = self.conv1(x, edge_index)
        x = self.leaky_relu(x)
        x = self.conv2(x, edge_index)
        x = self.leaky_relu(x)
        x = self.conv3(x, edge_index)
        x = self.leaky_relu(x)
        x = self.conv4(x, edge_index)
        x = self.leaky_relu(x)

        # 在进行全局最大池化之前确保 x 在维度 0 上具有非零大小
        if x.size(0) > 0:
            # 全连接层
            x = self.pooling(x, batch=None)
            x = self.fc1(x)
            x = self.leaky_relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.sigmoid(x)
        else:
            # 处理 x 沿维度 0 的大小为零的情况
            x = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)

        return x


class GCNClassifier_fc3(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, dropout_rate=0.5):
        super(GCNClassifier_fc3, self).__init__()

        # GCN 层
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, output_dim)

        # 图形池化层
        self.pooling = global_mean_pool

        # LeakyReLU 激活函数
        self.leaky_relu = nn.LeakyReLU()

        # 全连接层
        self.fc1 = nn.Linear(output_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # GCN 层
        x = self.conv1(x, edge_index)
        x = self.leaky_relu(x)
        x = self.conv2(x, edge_index)
        x = self.leaky_relu(x)
        x = self.conv3(x, edge_index)
        x = self.leaky_relu(x)
        x = self.conv4(x, edge_index)
        x = self.leaky_relu(x)

        # 在进行全局最大池化之前确保 x 在维度 0 上具有非零大小
        if x.size(0) > 0:
            # 全连接层
            x = self.pooling(x, batch=None)
            x = self.fc1(x)
            x = self.leaky_relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.leaky_relu(x)
            x = self.fc3(x)
            x = self.sigmoid(x)
        else:
            # 处理 x 沿维度 0 的大小为零的情况
            x = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)

        return x


if __name__ =='__main__':
    data_path='test_folder/1.php'
    graph_data = extract_graph_features(data_path)
    print(graph_data)