import torchvision
from matplotlib import pyplot as plt

from train_model import ResNet, ResidualBlock
import torch

if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,), (0.5,))])
    # 构建数据集
    path = 'E:/data'
    test_data = torchvision.datasets.MNIST(path, train=False, transform=transform)  # 测试集
    batch_size = 100  # Set batch size to 400 for 20 * 20 images

    testDataLoader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)  # Shuffle for randomness
    state_path = './model.pth'
    print('===> Loading weights : ' + state_path)
    net = torch.load(state_path)  # Load model state_dict instead of the whole model
    net.to(device)

    # 从测试集中选取一个batch做预测
    pred_test = enumerate(testDataLoader)
    batch_idx, (pred_data, pred_gt) = next(pred_test)
    pred_data = pred_data.to(device)
    output = net(pred_data)
    _, pred = torch.max(output.data, 1)  # 得到预测值
    pred = pred.cpu()
    pred_data = pred_data.cpu()

    print("ground truth: ", pred_gt)
    print("predict value: ", pred)

    fig = plt.figure(figsize=(10, 10))
    for i in range(batch_size):
        plt.subplot(10, 10, i + 1)
        plt.tight_layout()
        plt.imshow(pred_data[i][0], cmap='gray', interpolation='none')
        plt.title("GT:{} Pre: {}".format(pred_gt[i], pred[i]))
        plt.xticks([])
        plt.yticks([])

    plt.show()
