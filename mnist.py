import time
from model.vgg_autoencoder import VGGAutoEncoder,get_configs
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
# 数据加载和训练
def train_and_validate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 数据预处理
    transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 加载 MNIST 数据集
    train_dataset = MNIST(root='../data/Mnist', train=True, transform=transform, download=True)
    test_dataset = MNIST(root='../data/Mnist', train=False, transform=transform, download=True)

    # # 从训练集中随机抽取 4000 张数据
    # train_size = 2000
    # val_size = len(train_dataset) - train_size
    # train_dataset, _ = random_split(train_dataset, [train_size, val_size])
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    num_classes = 10
    samples_per_class = 1000
    # 获取每个类别的索引
    class_indices = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(train_dataset):
        class_indices[label].append(idx)
    # 从每个类别中随机抽取200张图像
    selected_indices = []
    for label in class_indices:
        selected_indices.extend(np.random.choice(class_indices[label], samples_per_class, replace=False))
    # 创建子数据集
    subset_dataset = Subset(train_dataset, selected_indices)
    # 创建DataLoader
    train_loader = DataLoader(subset_dataset, batch_size=64, shuffle=True)

    # 打印数据集大小
    print(f"Total number of images in the subset: {len(subset_dataset)}")

    # 模型初始化
    configs = get_configs('vgg16')
    model = VGGAutoEncoder(configs=configs).to(device)
    # 加载权重文件
    checkpoint = torch.load('../saved_models/imagenet-vgg16.pth')  # 确保路径正确
    state_dict = checkpoint['state_dict']  # 提取模型参数
    # 加载到模型
    model.load_state_dict(state_dict, strict=False)  # 设置 strict=False 以忽略多余的键
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # 训练
    num_epochs = 50

    last_model_path = "mnist_last_model1.pth"
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for images, _ in train_loader:
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}')

        # # 验证阶段
        # model.eval()
        # val_loss = 0
        # with torch.no_grad():
        #     for images, _ in test_loader:  # 使用测试集作为验证集
        #         images = images.to(torch.float32)
        #         outputs = model(images)
        #         loss = criterion(outputs, images)
        #         val_loss += loss.item()
        #
        # val_loss /= len(test_loader)
        # print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}')
        #
        # # 保存最佳模型
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     torch.save(model.state_dict(), best_model_path)
        #     print(f"Best model saved with validation loss: {best_val_loss:.4f}")

        # 保存最后一轮的模型
    torch.save(model.state_dict(), last_model_path)
    print(f"Last model saved to {last_model_path}")


# 调用训练和验证函数
start_time=time.time()
#train_and_validate()
end_time =time.time()
print("训练时间：",end_time-start_time)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 数据预处理
transform = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# 模型初始化
configs = get_configs('vgg16')
model = VGGAutoEncoder(configs=configs).to(device)
model_file = 'mnist_last_model1.pth'
model.load_state_dict(torch.load(model_file))
c = []
# 加载 MNIST 数据集
train_dataset = MNIST(root='../data/Mnist', train=True, transform=transform, download=True)
test_dataset = MNIST(root='../data/Mnist', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# 从训练集中随机抽取 4000 张数据
train_size = 2000
val_size = len(train_dataset) - train_size
train_dataset, _ = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
with torch.no_grad():  # 在评估过程中不需要计算梯度
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        c.append(outputs.cpu())

all_outputs = torch.cat(c, dim=0)
print(all_outputs.shape)
# 从 all_outputs 中取出前 100 个数据
sampled_outputs = all_outputs[:100]
# 将 100 个数据排列成 10x10 的网格
fig, axes = plt.subplots(10, 10, figsize=(15, 15))

# 遍历 sampled_outputs 并将它们显示在子图中
for i, ax in enumerate(axes.flat):
    # 将张量转换为 NumPy 数组并调整维度顺序
    output_image = sampled_outputs[i].permute(1, 2, 0).numpy()
    # 显示图像
    ax.imshow(output_image,cmap='gray')
    ax.axis('off')  # 隐藏坐标轴

# 调整子图间距
plt.subplots_adjust(wspace=0.1, hspace=0.1)
# 显示图像网格
plt.show()
