clc, clear;
%% 1. 加载Ionosphere数据集
load ionosphere
%% 使用PCA降维到3维
coeff = pca(trainData);
trainDataReduced = trainData * coeff(:,1:3);

% 绘制降维后的数据散点图
figure;
scatter3(trainDataReduced(trainLabels==1,1), trainDataReduced(trainLabels==1,2),...
    trainDataReduced(trainLabels==1,3), 'b', 'filled');
hold on;
scatter3(trainDataReduced(trainLabels==0,1), trainDataReduced(trainLabels==0,2),...
    trainDataReduced(trainLabels==0,3), 'r', 'filled');
title('Ionosphere数据集 PCA降维后的三维散点图');
xlabel('主成分 1');
ylabel('主成分 2');
zlabel('主成分 3');
legend('正类', '负类');
hold off;

%% 2. 数据预处理
X = Ionosphere(:, 2:end);
Y = Ionosphere(:, 1);

% 划分数据集
trainData = X(1:340, 2:end);
trainLabels = Y(1:340);
testData = X(341:end, 2:end);
testLabels = Y(341:end);

%% 不同核函数类型和参数
% 定义不同的核函数
kernelTypes = {'linear', 'polynomial', 'rbf'};

% 更多取点的C值
C_values = [0.1, 0.5, 1, 5, 10, 50];

% 存储不同核函数和不同C值下的准确率
accuracies = zeros(length(kernelTypes), length(C_values));

% 比较不同核函数和不同C值
for k = 1:length(kernelTypes)
    for i = 1:length(C_values)
        % 使用每种核函数和C值训练SVM模型
        SVMModel = fitcsvm(trainData, trainLabels, 'KernelFunction',...
            kernelTypes{k}, 'BoxConstraint', C_values(i));
        
        % 预测
        predictedLabels = predict(SVMModel, testData);
        
        % 计算分类准确率
        accuracy = sum(predictedLabels == testLabels) / numel(testLabels);
        accuracies(k, i) = accuracy;
    end
end

% 可视化结果
figure;
bar(accuracies);
xlabel('核函数类型');
ylabel('准确率');
set(gca, 'xticklabel', kernelTypes);
legendCell = cellstr(num2str(C_values', 'C = %-g'));
legend(legendCell);
title('SVM核函数和不同C值的比较');



