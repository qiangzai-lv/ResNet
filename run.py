import torch
import torchvision
import argparse

from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet34
from torchvision.transforms import InterpolationMode


def train(args):
    # 超参数设置，方便管理
    num_epochs = args.max_epoch
    batch_size = args.batch_size
    learning_rate = args.lr
    image_size = args.image_size
    momentum = args.momentum
    # gpu
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    # CIFAR10数据集加载

    # 设置数据集的格式
    transform = transforms.Compose([transforms.Resize((image_size, image_size),
                                                      interpolation=InterpolationMode.BICUBIC),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.2435, 0.2616])
                                    ])

    # 数据加载
    # 如果没有这个数据集的话会自动下载
    train_data = torchvision.datasets.CIFAR10(root="dataset", download=True, transform=transform, train=True)
    test_data = torchvision.datasets.CIFAR10(root="dataset", download=True, transform=transform, train=False)
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    print('Data load is Ready')
    # 添加tensorboard路径
    model = resnet34(num_classes=10).to(device)
    # 参数量估计
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.2fM" % (total / 1e6))
    # Loss and optimizer
    # 选择交叉熵作为损失函数
    criterion = nn.CrossEntropyLoss()
    # 选择SGD为优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    total_train_step = 0  # 记录训练次数
    total_test_step = 0  # 记录测试次数
    # 开始训练
    for epoch in range(num_epochs):
        print("---------------第{}轮训练开始-------------".format(epoch + 1))
        for i, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_step = total_train_step + 1
            if (i + 1) % args.print_map_epoch == 0:  # 100次显示一次loss
                print("Epoch [{}/{}], Step [{}] Loss: {:.4f}"
                      .format(epoch + 1, num_epochs, total_train_step, loss.item()))

        # Test the model
        model.eval()
        total_test_loss = 0
        total_accuracy = 0  # 正确率
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                total_test_loss += loss
                total_accuracy += correct
            print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
            total_test_step += 1
    # Save the model checkpoint
    torch.save(model, 'weights.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    '''------------------------------------------调  节  部  分------------------------------------------------------'''
    parser.add_argument('--max_epoch', type=int, default=30, help='total epoch')
    parser.add_argument('--device_num', type=str, default='cpu', help='select GPU or cpu')
    parser.add_argument('--image_size', type=int, default=224, help='if crop img, img will be resized to the size')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size, recommended 16')
    parser.add_argument('--lr', type=float, default=0.007, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.90, help='choice a float in 0-1')
    parser.add_argument('--print_map_epoch', type=int, default=100, help='')
    parser.add_argument('--SummerWriter_log', type=str, default='Resnet', help='')

    args = parser.parse_args()

    train(args)
