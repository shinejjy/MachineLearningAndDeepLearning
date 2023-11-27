import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true])
    loss = np.sum(log_likelihood) / m
    return loss


def one_hot_encode(y, num_classes):
    m = y.shape[0]
    one_hot = np.zeros((m, num_classes))
    one_hot[np.arange(m), y] = 1
    return one_hot


def train_neural_network(X, y, hidden_layer_sizes, learning_rate, epochs):
    input_size = X.shape[1]
    output_size = len(np.unique(y))
    num_layers = len(hidden_layer_sizes) + 1

    # 初始化权重和偏差
    weights = [np.random.randn(input_size, hidden_layer_sizes[0])]
    biases = [np.zeros((1, hidden_layer_sizes[0]))]
    for i in range(1, num_layers - 1):
        weights.append(np.random.randn(hidden_layer_sizes[i - 1], hidden_layer_sizes[i]))
        biases.append(np.zeros((1, hidden_layer_sizes[i])))
    weights.append(np.random.randn(hidden_layer_sizes[-1], output_size))
    biases.append(np.zeros((1, output_size)))

    for epoch in range(epochs):
        # 前向传播
        layer_outputs = [X]
        for i in range(num_layers):
            layer_inputs = layer_outputs[-1] @ weights[i] + biases[i]
            if i == num_layers - 1:
                layer_outputs.append(softmax(layer_inputs))
            else:
                layer_outputs.append(sigmoid(layer_inputs))

        # 计算损失
        loss = cross_entropy_loss(y, layer_outputs[-1])

        # 反向传播
        errors = [layer_outputs[-1] - one_hot_encode(y, output_size)]
        for i in range(num_layers - 2, -1, -1):
            errors.insert(0, errors[0] @ weights[i + 1].T * sigmoid_derivative(layer_outputs[i + 1]))

        # 更新权重和偏差
        for i in range(num_layers):
            weights[i] -= learning_rate * layer_outputs[i].T @ errors[i]
            biases[i] -= learning_rate * np.sum(errors[i], axis=0, keepdims=True)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return weights, biases


def predict(X, weights, biases):
    num_layers = len(weights)
    layer_outputs = [X]
    for i in range(num_layers):
        layer_inputs = layer_outputs[-1] @ weights[i] + biases[i]
        if i == num_layers - 1:
            layer_outputs.append(softmax(layer_inputs))
        else:
            layer_outputs.append(sigmoid(layer_inputs))
    return np.argmax(layer_outputs[-1], axis=1)


# 数据准备
# 加载Iris数据集
iris = load_iris()
X, y = iris.data, iris.target

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 设置超参数
learning_rate = 0.01
epochs = 1000
hidden_layer_sizes = [10]  # 调整隐藏层的数量和神经元数量

# 10次10折交叉验证
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

accuracies = []

for idx, (train_index, test_index) in enumerate(kf.split(X, y)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # 训练神经网络
    weights, biases = train_neural_network(X_train, y_train, hidden_layer_sizes, learning_rate, epochs)

    # 预测
    predictions = predict(X_test, weights, biases)

    # 计算准确率
    accuracy = np.mean(predictions == y_test)
    print(f"Fold {idx + 1} Accuracy: {accuracy: .2%}\n")
    accuracies.append(accuracy)

# 打印平均准确率
average_accuracy = np.mean(accuracies)
print(f"Average Accuracy: {average_accuracy:.2%}")