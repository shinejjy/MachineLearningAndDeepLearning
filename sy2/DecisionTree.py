import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

# 设置合适的字体
plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文显示字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题


class TreeNode:
    def __init__(self, parent=None, feature=None, value=None, text='None', index=None):
        self.children = {}
        self.parent = parent
        self.feature = feature
        self.value = value
        self.text = text
        self.index = index


import numpy as np


class TreeNode:
    def __init__(self, parent=None, feature=None, feature_index=None, value=None, index=None, text=None):
        self.parent = parent
        self.feature = feature
        self.feature_index = feature_index
        self.value = value
        self.index = index
        self.text = text
        self.children = {}


class DecisionTree:
    def __init__(self, feature_name, mapping, max_depth=None):
        self.max_depth = max_depth
        self.index = 0
        self.feature_name = feature_name
        self.mapping = mapping

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

        unique_values = np.unique(X[:, best_feature])
        tree = TreeNode(parent=parent, feature=self.feature_name[best_feature], feature_index=best_feature,
                        text=text, index=str(self.index))
        self.index += 1

        for value in unique_values:
            branch_indices = X[:, best_feature] == value
            branch_X = X[branch_indices]
            branch_y = y[branch_indices]
            feature = self.feature_name[best_feature]
            tree.children[value] = self.fit(branch_X, branch_y, parent=tree, depth=depth + 1,
                                            text=f'{self.mapping[feature][value]}', criterion=criterion)

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

            if weighted_gini <= best_gini:
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

            if gain_ratio >= best_gain_ratio:
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
        result = []
        for x in X:
            result.append(self.predict_item(x, tree))

        return result

    def predict_item(self, x, tree):
        if tree.value is not None:
            return tree.value
        else:
            child_tree = tree.children[x[tree.feature_index]]
            return self.predict_item(x, child_tree)


class PlotDT:
    def __init__(self):
        self.G = nx.DiGraph()

    def plotNode(self, tree, depth):
        if tree.children:
            self.G.add_node(tree.index, label=tree.feature + '=？', partition=depth)
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
