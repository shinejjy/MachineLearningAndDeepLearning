import networkx as nx
import numpy as np
import pandas as pd
from graphviz import Digraph
from matplotlib import pyplot as plt
import matplotlib

# 设置合适的字体
plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文显示字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题


def entropy(y):
    unique_labels, label_counts = np.unique(y, return_counts=True)
    probs = label_counts / len(y)
    entropy = -np.sum(probs * np.log2(probs))
    return entropy


def information_gain(X, y, feature):
    unique_values = np.unique(X[feature])
    total_entropy = entropy(y)
    weighted_entropy = 0

    for value in unique_values:
        subset_y = y[X[feature] == value]
        weighted_entropy += len(subset_y) / len(y) * entropy(subset_y)

    information_gain = total_entropy - weighted_entropy
    return information_gain


def gain_ratio(X, y, feature):
    unique_values = np.unique(X[feature])
    total_entropy = entropy(y)
    split_information = 0
    weighted_entropy = 0

    for value in unique_values:
        subset_y = y[X[feature] == value]
        split_information -= len(subset_y) / len(y) * np.log2(len(subset_y) / len(y))
        weighted_entropy += len(subset_y) / len(y) * entropy(subset_y)

    gain = total_entropy - weighted_entropy
    if split_information == 0:
        gain_ratio = 0
    else:
        gain_ratio = gain / split_information

    return gain_ratio


def gini_index(y):
    unique_labels, label_counts = np.unique(y, return_counts=True)
    probs = (label_counts / len(y)) ** 2
    gini = 1 - np.sum(probs)
    return gini


def gini_index_feature(X, y, feature):
    unique_values = np.unique(X[feature])
    weighted_gini = 0

    for value in unique_values:
        subset_y = y[X[feature] == value]
        weighted_gini += len(subset_y) / len(y) * gini_index(subset_y)

    return weighted_gini


class TreeNode:
    def __init__(self, parent=None, feature=None, value=None, text='None', index=None):
        self.pos = None
        self.children = {}
        self.parent = parent
        self.feature = feature
        self.value = value
        self.text = text
        self.index = index


class DecisionTree:
    def __init__(self, feature_name, max_depth=None):
        self.max_depth = max_depth
        self.index = 0
        self.feature_name = feature_name

    def fit(self, X, y, parent=None, depth=0, text='None', criterion='entropy'):
        if depth == self.max_depth or len(set(y)) == 1:
            # 如果达到最大深度或者标签唯一，停止分裂
            leaf_node = TreeNode(parent=parent, value=np.argmax(np.bincount(y)), index=str(self.index), text=text)
            self.index += 1
            return leaf_node

        if criterion == 'entropy':
            best_feature = self.select_best_feature(X, y)
        elif criterion == 'gini':
            best_feature = self.select_best_feature_gini(X, y)
        elif criterion == 'gain_ratio':
            best_feature = self.select_best_feature_gain_ratio(X, y)
        else:
            raise ValueError("Invalid criterion. Use 'entropy', 'gini', or 'gain_ratio'.")

        if best_feature is None:
            return TreeNode(parent=parent, value=np.argmax(np.bincount(y)))

        unique_values = np.unique(X[:, best_feature])
        tree = TreeNode(parent=parent, feature=self.feature_name[best_feature], text=text, index=str(self.index))
        self.index += 1

        for value in unique_values:
            branch_indices = X[:, best_feature] == value
            branch_X = X[branch_indices]
            branch_y = y[branch_indices]
            tree.children[value] = self.fit(branch_X, branch_y, parent=tree, depth=depth + 1,
                                           text=f'{self.feature_name[best_feature]} = {value}', criterion=criterion)

        return tree

    def select_best_feature(self, X, y):
        best_information_gain = -1
        best_feature = None
        total_entropy = self.entropy(y)

        for feature in range(X.shape[1]):
            unique_values = np.unique(X[:, feature])
            weighted_entropy = 0

            for value in unique_values:
                subset_indices = X[:, feature] == value
                subset_y = y[subset_indices]
                weighted_entropy += len(subset_y) / len(y) * self.entropy(subset_y)

            information_gain = total_entropy - weighted_entropy

            if information_gain >= best_information_gain:
                best_information_gain = information_gain
                best_feature = feature

        return best_feature

    def select_best_feature_gini(self, X, y):
        best_gini = float('inf')
        best_feature = None

        for feature in range(X.shape[1]):
            unique_values = np.unique(X[:, feature])
            weighted_gini = 0

            for value in unique_values:
                subset_indices = X[:, feature] == value
                subset_y = y[subset_indices]
                weighted_gini += len(subset_y) / len(y) * self.gini(subset_y)

            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = feature

        return best_feature

    def select_best_feature_gain_ratio(self, X, y):
        best_gain_ratio = -1
        best_feature = None
        total_entropy = self.entropy(y)
        total_samples = len(y)

        for feature in range(X.shape[1]):
            unique_values = np.unique(X[:, feature])
            weighted_entropy = 0
            split_info = 0

            for value in unique_values:
                subset_indices = X[:, feature] == value
                subset_y = y[subset_indices]
                weighted_entropy += (len(subset_y) / total_samples) * self.entropy(subset_y)
                split_info -= (len(subset_y) / total_samples) * np.log2(len(subset_y) / total_samples)

            information_gain = total_entropy - weighted_entropy
            gain_ratio = information_gain / (split_info + 1e-10)  # Add a small value to avoid division by zero

            if gain_ratio > best_gain_ratio:
                best_gain_ratio = gain_ratio
                best_feature = feature

        return best_feature

    def gini(self, y):
        unique_labels, label_counts = np.unique(y, return_counts=True)
        probs = label_counts / len(y)
        gini_index = 1 - np.sum(probs ** 2)
        return gini_index


    def entropy(self, y):
        unique_labels, label_counts = np.unique(y, return_counts=True)
        probs = label_counts / len(y)
        entropy = -np.sum(probs * np.log2(probs))
        return entropy

    def predict(self, X, tree):
        if tree.value:
            return tree.value
        else:
            child_tree = tree.children[X[tree.feature]]
            return self.predict(X, child_tree)


data_train = pd.read_csv('watermelon_train.txt', delimiter=',', header=0)
data_test = pd.read_csv('watermelon_test.txt', delimiter=',', header=0)
pre_data = pd.concat([data_train, data_test], ignore_index=True)


def auto_encode_categorical_features(data):
    mapping_dict = {}
    re_mapping_dict = {}
    for index, column in enumerate(data.select_dtypes(include=['object']).columns):
        unique_values = data[column].unique()
        mapping = {value: idx for idx, value in enumerate(unique_values)}
        re_mapping = {idx: value for idx, value in enumerate(unique_values)}
        data[column] = data[column].map(mapping)
        mapping_dict[column] = mapping
        re_mapping_dict[index] = re_mapping

    return data, mapping_dict, re_mapping_dict


# 调用函数进行自适应编码
data, mapping_dict, re_mapping_dict = auto_encode_categorical_features(pre_data)

X_train = data.iloc[0:10, 1:7].to_numpy()  # 选择前10行的特征列
y_train = data.iloc[0:10, 7].to_numpy()  # 选择前10行的标签列
X_test = data.iloc[10:17, 1:7].to_numpy()  # 选择前7行的特征列
y_test = data.iloc[10:17, 7].to_numpy()  # 选择前7行的标签列

tree = DecisionTree(pre_data.columns[1: 7], max_depth=None)
decision_tree = tree.fit(X_train, y_train)
print(decision_tree)

import networkx as nx
import matplotlib.pyplot as plt


class PlotDT:
    def __init__(self):
        self.G = nx.DiGraph()

    def plotNode(self, tree, depth):
        if tree.children:
            self.G.add_node(tree.index, label=tree.feature + '？', partition=depth)
        else:
            self.G.add_node(tree.index, label='好瓜' if tree.value == 0 else '坏瓜', partition=depth)

    def plotEdge(self, tree):
        self.G.add_edge(tree.parent.index, tree.index, label=tree.text)

    def plotTree(self, tree):
        self.creatPlot(tree, depth=0)
        pos = self.get_node_positions()

        edge_labels = {(source, target): data['label'] for source, target, data in self.G.edges(data=True)}
        node_labels = {node: data['label'] for node, data in self.G.nodes(data=True)}

        nx.draw(self.G, pos, labels=node_labels, with_labels=True, node_size=1000, node_color='lightblue')
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, font_color='red')
        plt.show()

    def creatPlot(self, tree, depth):
        if tree.parent is not None:
            self.plotEdge(tree)
        self.plotNode(tree, depth)
        if tree.children:
            for _, child_tree in tree.children.items():
                self.creatPlot(child_tree, depth + 1)

    def get_node_positions(self):
        return nx.multipartite_layout(self.G, subset_key="partition")


# Example usage:
# plotter = PlotDT()
# plotter.plotTree(root_node)


canvas = PlotDT()
canvas.plotTree(decision_tree)
