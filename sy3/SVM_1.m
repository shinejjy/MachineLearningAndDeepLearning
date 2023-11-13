clc,clear;
%% 生成两组二维随机正态分布数据
rng(1); % 设置随机数种子以保证结果可复现
mu1 = [1 1];
sigma1 = [1.5 0.5; 0.5 1.5];
data1 = mvnrnd(mu1, sigma1, 100); % 第一组数据

mu2 = [5 6];
sigma2 = [1 0.5; 0.5 1];
data2 = mvnrnd(mu2, sigma2, 100); % 第二组数据

% 创建标签
group1 = ones(100, 1); % 标签为1
group2 = ones(100, 1) * -1; % 标签为-1

% 合并数据和标签
data = [data1; data2];
groups = [group1; group2];

figure;
h = gscatter(data(:,1), data(:,2), groups);
title('样本可视化');
xlabel('特征 1');
ylabel('特征 2');
legend('数据集 1', '数据集 2');

%% 训练SVM模型和二次规划
SVMModel = fitcsvm(data, groups, 'KernelFunction', 'linear', 'BoxConstraint', Inf);
% 执行线性回归
mdl = fitlm(data, groups);

%% 绘制数据散点图和SVM决策边界与最大间隔
figure;
h = gscatter(data(:,1), data(:,2), groups);
hold on;

% 绘制SVM决策边界和最大间隔
w = SVMModel.Beta;
b = SVMModel.Bias;

% 计算支持向量
sv = SVMModel.SupportVectors;

% 计算SVM决策边界
f = @(x) (w(1)*x + b)/(-w(2));
fplot(f, [min(data(:,1)) max(data(:,1))], 'k--');

% 获取线性回归系数
coefficients = mdl.Coefficients.Estimate;
b0 = coefficients(1);
b1 = coefficients(2:end);

% 绘制线性回归线
f_linear = @(x) (b0 + b1(1)*x)/(-b1(2));
fplot(f_linear, [min(data(:,1)) max(data(:,1))], 'm');

% 绘制支持向量
plot(sv(:,1), sv(:,2), 'ko', 'MarkerSize', 10);

% 计算最大间隔上边界
f_upper = @(x) (w(1)*x + b + 1)/(-w(2));
fplot(f_upper, [min(data(:,1)) max(data(:,1))], 'k--');

% 计算最大间隔下边界
f_lower = @(x) (w(1)*x + b - 1)/(-w(2));
fplot(f_lower, [min(data(:,1)) max(data(:,1))], 'k--');

hold off;

title('SVM与线性回归可视化');
xlabel('特征 1');
ylabel('特征 2');
legend('数据集 1', '数据集 2', 'SVM', '二次规划', '支持向量');

