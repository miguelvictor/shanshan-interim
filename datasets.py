from itertools import chain
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm

import json
import pickle
import torch


class ShanshanDataset(Dataset):
    def __init__(self, key='train', dataset_dir='dataset'):
        dataset_path = Path(dataset_dir)
        self.init_global_labels(dataset_path)
        self.items = list((dataset_path / key).rglob('*.json'))
        assert len(self.items) != 0, 'No json files found.'

    def init_global_labels(self, dataset_path: Path, cache_output_path: str = '__global_labels.pkl'):
        # detect cache file existence
        cache_output_path = Path(cache_output_path)
        if cache_output_path.is_file():
            with cache_output_path.open('rb') as fd:
                self.global_labels = pickle.load(fd)
                self.n_labels = len(self.global_labels)
                return

        train_graphs = Path(dataset_path / 'train').rglob('*.json')
        dev_graphs = Path(dataset_path / 'dev').rglob('*.json')
        graphs = list(chain(train_graphs, dev_graphs))
        assert len(graphs) != 0, 'No json files found.'
        global_labels = set()

        for fp in tqdm(graphs):
            with fp.open('r') as fd:
                data = json.load(fd)

            global_labels = global_labels.union(set(data["nodes_label1"]))
            global_labels = global_labels.union(set(data["nodes_label2"]))

        # global_labels是一个字典，node文本 -> index 的映射
        # global_labels就是一个语料库
        self.global_labels = {v: i for i, v in enumerate(global_labels)}
        self.n_labels = len(self.global_labels)

        # pickle object for subsequent usage
        with cache_output_path.open('wb') as fd:
            pickle.dump(self.global_labels, fd)

    def transfer_to_torch(self, data):
        """
        Transferring the data to torch and creating a hash table.
        Including the indices, features and target.

        :param data: Data dictionary.
        :return new_data: Dictionary of Torch Tensors.
        """
        # 转换图为无向图
        # 边记为 e ， 则edges_1的尺寸为 2e * 2
        edges_1 = data["graph1"] + [[y, x] for x, y in data["graph1"]]
        edges_2 = data["graph2"] + [[y, x] for x, y in data["graph2"]]

        # # edges_1_T的尺寸为 2* 2e; 上面的行是出发顶点；下面的行是目标顶点
        edges_1 = torch.tensor(edges_1, dtype=torch.long).transpose(0, 1)
        edges_2 = torch.tensor(edges_2, dtype=torch.long).transpose(0, 1)

        features_1, features_2 = [], []

        # features_1 的尺寸是 len(data["labels_1"]) * len(self.global_labels.values())
        # 即： [num_nodes, all_num_features]
        # 用一个矩阵记录第i个节点的内容是什么
        # one-hot编码
        for text1 in data["nodes_label1"]:
            features_1.append([
                1 if self.global_labels[text1] == i else 0
                for i in self.global_labels.values()
            ])

        for text2 in data["nodes_label2"]:
            features_2.append([
                1 if self.global_labels[text2] == i else 0
                for i in self.global_labels.values()
            ])

        # convert python arrays to pytorch tensors
        features_1 = torch.tensor(features_1, dtype=torch.float)
        features_2 = torch.tensor(features_2, dtype=torch.float)

        return {
            'edge_index_1': edges_1,
            'edge_index_2': edges_2,
            'features_1': features_1,
            'features_2': features_2,
            'target': int(data['label']),
        }

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        with self.items[idx].open('r') as fd:
            data = json.load(fd)
            data = self.transfer_to_torch(data)
            return data
