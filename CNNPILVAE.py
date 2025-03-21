import warnings
import numpy as np
warnings.filterwarnings("ignore")
from model.vgg_autoencoder import VGGAutoEncoder,get_configs
import torch
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from scipy.linalg import orth
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import os
from torchvision import datasets, transforms
from  model.UNET import UNET_AutoEncoder
from model.GAI_model2 import CovNet
from model.model import Autoencoder
import time
hidden_size = [2000]  # 预定义隐藏层神经元数量
latent_dim = 16  # dim of Z   8
MAX_EN_LAYER = len(hidden_size)
mean_value = 0
sig = 1e-2
para = 1
actFun = 'prelu'  # prelu gau
lambda0 = 1e-4  # 1e-6
vae = []
HiddenO = []
l = 1
batch_size=64
value_to_find =9 #查找的索引值
num_training_samples = 2000 # 训练样本量

transform=transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data/Mnist', train=True, download=True,
                   transform=transform),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data/Mnist', train=False, transform=transform),
    batch_size=batch_size, shuffle=False)
# 遍历 DataLoader 中的每一个 batch
target_data=[]

for batch_data, batch_label in train_loader:
    
    mask = torch.eq(batch_label, value_to_find)

    indices = torch.nonzero(mask).squeeze()

    target_batch_data = torch.index_select(batch_data, 0, indices)

    # 将每一个 batch 的指定标签的数据添加到结果列表中
    target_data.append(target_batch_data)
# 拼接所有的数据
target_data = torch.cat(target_data, dim=0)

if target_data.size(0) >= num_training_samples:
    # 创建随机索引
    indices = torch.randperm(target_data.size(0))[:num_training_samples]
    # 根据索引抽取数据
    target_data = target_data[indices]
print("target_data.shape",target_data.shape)
#model=Autoencoder()
#model.load_state_dict(torch.load(f'../saved_models/MnistModel/SimpleAE/COV-32/model_epoch_96.pth'))
model = VGGAutoEncoder(get_configs("vgg16"))
model.load_state_dict(torch.load(f'../train_vgg/mnist_last_model1.pth'))

model.eval()  
with torch.no_grad(): 
    encoder_output =model.encoder(target_data)
    print("经过卷积之后的形状",encoder_output.shape)



selected_images = encoder_output.numpy()
# 打印结果
print("选中的图像形状:", selected_images.shape)

X_train = selected_images

X_train_2d = X_train.reshape(X_train.shape[0], -1) 
print("原始三维数组形状:", X_train.shape)
print("合并后的二维数组形状:", X_train_2d.shape)
rank_data = np.linalg.matrix_rank(X_train_2d)
print("输入的秩：", rank_data)
input_dim = X_train_2d.shape[1]  # dim of input data
hidden_dim = X_train_2d.shape[0]  # Gn-PIL dim of H

InputLayer = X_train_2d.T
print("InputLayer's shape:", InputLayer.shape)


def ActivationFunc(tempH, ActivationFunction, p):#激活函数
    if ActivationFunction == 'relu':
        #         tempH[tempH <= 0] = 0
        #         tempH[tempH > 0] = tempH
        #         H = tempH
        H = np.maximum(0, tempH)
    elif ActivationFunction == 'prelu':
        alpha = 0.02;
        #         tempH[tempH <= 0] = alpha*tempH;
        #         H = tempH
        H = np.maximum(alpha * tempH, tempH)
    elif ActivationFunction == 'gelu':  # xσ(1.702x)
        H = tempH * 1.0 / (1 + np.exp(-p * tempH * 1.702))
    elif ActivationFunction == 'sigmoid':
        H = 1.0 / (1 + np.exp(-tempH))
    elif ActivationFunction == 'srelu':
        tempH[tempH <= 0] = 0
        tempH[tempH > 0] = tempH
        H = tempH
    elif ActivationFunction == 'sin':
        H = np.sin(tempH)
    elif ActivationFunction == 'tanh':
        H = np.np.tanh(tempH)
    return H





def Gai_PIL0(InputLayer, input_dim, hidden_dim, layer_idx):

    InputWeight = np.random.randn(hidden_dim, input_dim)
    if hidden_dim >= input_dim:
        InputWeight = orth(InputWeight)
    else:
        InputWeight = orth(InputWeight.T).T
    # Compute the rank of the matrix InputLayer

    print("Inputweight的形状", InputWeight.shape)
    matrix_rank = np.linalg.matrix_rank(InputLayer)

    tempH = InputWeight.dot(InputLayer)
    H1 = ActivationFunc(tempH, actFun, para)

    
    layer_idx = layer_idx + 1
    InputLayer = H1
    hidden_dim = InputLayer.shape[1]
    input_dim = InputLayer.shape[0]
    
    vae.append(InputWeight)  # vae{l}.WI = InputWeight
    HiddenO.append(H1)
    return InputLayer, input_dim, hidden_dim, layer_idx


def Gn_PIL(InputLayer, l):
    InputLayer_pinv = np.linalg.pinv(InputLayer)

    # Compute the rank of the matrix InputLayer
    matrix_rank = np.linalg.matrix_rank(InputLayer_pinv)
    # 生成具有指定均值和标准差的随机数，大小与 InputLayer_pinv 相同
    random_noise = np.random.normal(mean_value, sig, size=InputLayer_pinv.shape)

    # 将随机噪声添加到 InputLayer_pinv 中
    InputLayer_pinv = InputLayer_pinv + random_noise
    tempH = InputLayer_pinv.dot(InputLayer)
    H2 = ActivationFunc(tempH, actFun, para)

    hidden_size =H2.shape[0]
    num = H2.shape[1]
    l = l + 1
    vae.append(InputLayer_pinv)
    HiddenO.append(H2)
    return H2, l, num, hidden_size

def ppcamle(data  , q):
    N, d = data.shape        
    mu = np.mean(data, axis=0)
    T = data - np.tile(mu, (N, 1))  # T = data - repmat(mu, N, 1)
    S = T.T.dot(T) / N  # S = T' * T / N
    D, V = np.linalg.eig(S)  # Eigenvalue decomposition
    sorted_indices = np.argsort(D)[::-1]  
    D = D[sorted_indices] 
    V = V[:, sorted_indices]
    sigma = np.sum(D[(q + 1):]) / (d - q) 
    Uq = V[:, :q]
    lambda_q = D[:q]
    w = Uq.dot(np.sqrt(np.diag(lambda_q) - sigma * np.identity(q))) 
    C = w.dot(w.T) + sigma * np.identity(d)
    L = -N * (d * np.log(2 * np.pi) + np.log(np.linalg.det(C)) + np.trace(np.linalg.solve(C, S)) )/ 2 
    return mu, w, sigma, L
def ppca(H2, l, latent_dim=latent_dim):
    mu, w, sigma, L = ppcamle(H2, latent_dim)
    print(w.shape)#(1000, 8)
    a=np.linalg.solve(w.T.dot(w) + sigma * np.identity(latent_dim), w.T)
    print(a.shape)
    # 计算 Z  Z 是一个 (latent_dim, num_trainingSamples) 的矩阵
    # Z = np.linalg.solve(w.T.dot(w) + sigma * np.identity(latent_dim), w.T).dot(
    #     H2 - np.tile(mu.T, (num_training_samples, 1)).T)
    Z = np.linalg.solve(w.T.dot(w) + sigma * np.identity(latent_dim), w.T).dot(
        H2 - np.tile(mu.T, (num_training_samples, 1)).T)
    vae.append(w)
    l = l + 1
    return Z, l, sigma, w, mu, L

def Zrec(Z, H2, l, lambda0=lambda0):#用于重构潜在空间中的数据
    ZZT = Z.dot(Z.T)
    OutputWeight = H2.dot(Z.T).dot(np.linalg.pinv(ZZT + lambda0 * np.eye(latent_dim)))
    vae.append(OutputWeight)
    tempH = OutputWeight.dot(Z)
    l = l + 1
    return tempH, l


def H2rec(tempH, l, dl,hidden_dim,lambda0=lambda0 ):
    tempH_transpose = tempH.T

    tempHHT = tempH.dot(tempH_transpose)

    eye_matrix = lambda0 * np.eye(hidden_dim)#(1000, 1000)

    #     OutputWeight = HiddenO[dl - 1].dot(tempH_transpose) / (tempHHT + eye_matrix)
    OutputWeight = (HiddenO[dl - 1].dot(tempH_transpose)).dot(np.linalg.pinv(tempHHT + eye_matrix))
    vae.append(OutputWeight)  # 5

    # 更新 tempH
    tempH = OutputWeight.dot(tempH)
    l += 1
    dl -= 1
    return tempH, l, dl

def H1rec(rec_H1, X_train_2d, lambda0=lambda0):
    rec_H1H1T = rec_H1.dot(rec_H1.T)
    lambda_eye = lambda0 * np.eye(hidden_dim)

    OutputWeight = (X_train_2d.T.dot(rec_H1.T)).dot(np.linalg.pinv(rec_H1H1T + lambda_eye))
    vae.append(OutputWeight)  # 6

    # 计算 tempX
    tempH = OutputWeight.dot(rec_H1)
    return tempH

start_time = time.time()  
for hidden in hidden_size:
    InputLayer, input_dim, hidden_dim, l = Gai_PIL0(InputLayer, input_dim, hidden, l)
    print(f"Layer {l}: shape = {InputLayer.shape}")

InputLayer, l ,input_dim,hidden_dim= Gn_PIL(InputLayer, l)#l=1
print("H1",InputLayer.shape,"l",l,"input_dim",input_dim)



Y = InputLayer.T #(100, 1000)
print("Y",Y.shape)#(1000, 1000)
Z, l, sigma,w,mu,L = ppca(Y, l)
print("降维后的形状",Z.shape)  #torch.Size([15, 5851])


rec_H2, l = Zrec(Z, InputLayer, l)
print(rec_H2.shape)
print(l)#6

for hidden in HiddenO:
    print("hidden",hidden.shape)

dl = l - 4 # 2
while dl > 0:
    rec_H2, l, dl = H2rec(rec_H2, l,dl,hidden_dim)
    hidden_dim=hidden_size[dl]
    print(1)

rec_H1=rec_H2
print(rec_H1.shape)
rec_X = H1rec(rec_H1, X_train_2d)
# 更新 rec_X
print(rec_X.shape)
end_time=time.time()
print("训练时间：",end_time-start_time)
rec_X=rec_X.T.reshape(encoder_output.shape)
print(rec_X.shape)

if isinstance(rec_X, np.ndarray):  
    rec_X = torch.from_numpy(rec_X).float() 

with torch.no_grad():  # 在评估过程中不需要计算梯度
    rec_X=model.decoder(rec_X)


fig, axes = plt.subplots(10, 10, figsize=(15, 15))


for i, ax in enumerate(axes.flat):
    # 将张量转换为 NumPy 数组并调整维度顺序
    output_image = rec_X[i].permute(1, 2, 0).numpy()
    # 显示图像
    ax.imshow(output_image,cmap='gray')
    ax.axis('off')  # 隐藏坐标轴

# 调整子图间距
plt.subplots_adjust(wspace=0.1, hspace=0.1)
# 显示图像网格
plt.show()




# Generating new sample
# num_samples = 1000
# latent_samples = np.random.randn(latent_dim, num_samples)
# # Generate the covariance matrix
# print(sigma)
# covariance = sigma * np.eye(hidden_dim)
# # Generate noise from the Gaussian distribution
# # ns = mvnrnd(zeros(num_samples, hidden_dim), covariance, num_samples);
# ns = multivariate_normal.rvs(mean=np.zeros(hidden_dim), cov=covariance, size=num_samples)
# print("ns's shape:", ns.shape)
# # Generate new samples
# hidden_samples1 = w.dot(latent_samples) + np.tile(mu.T, (num_samples, 1)).T
# l = 4 + MAX_EN_LAYER
# generated_samples2 = vae[l - 1].dot(hidden_samples1)
# temprecH = generated_samples2
# while l < 4 + MAX_EN_LAYER * 2:
#     l = l + 1
#     temprecH = vae[l - 1].dot(temprecH)
#
# generated_X = temprecH.T
# print("generated_X:", generated_X.shape)
# generated_X=generated_X.reshape(num_samples,encoder_output.shape[1],encoder_output.shape[2],encoder_output.shape[3])
# if isinstance(generated_X, np.ndarray):  # 检查是否为 NumPy 数组
#     generated_X = torch.from_numpy(generated_X).float()  # 转换为 PyTorch 张量并指定数据类型
# with torch.no_grad():  # 在评估过程中不需要计算梯度
#     generated_X=model.decoder(generated_X)
# #     遍历每张图片并将其添加到图形子集中
# for i in range(100):
#     plt.subplot(10, 10, i + 1)
#     output_image = generated_X[i].permute(1, 2, 0).numpy()
#     plt.imshow(output_image, cmap='gray')  # 假设这是灰度图像
#     plt.axis('off')  # 关闭坐标轴
# # 显示图形
# plt.show()
#
# # 假设 generated_X 是你生成的图像数组，形状为 (num_images, height, width)
# num_images = generated_X.shape[0]  # 例如, 100 张图片
# # Step 1: Resize to (28, 28)
# generated_X_resized = F.interpolate(generated_X, size=(28, 28), mode='bilinear', align_corners=False)
#
# # Step 2: Convert to grayscale by averaging channels
# generated_X_grayscale = generated_X_resized.mean(dim=1, keepdim=True)  # 形状变为 (1000, 1, 28, 28)
#
# # 转换为 NumPy 数组（假设后续需要 NumPy 格式）
# generated_X_grayscale_np = generated_X_grayscale.squeeze(1).cpu().numpy()  # 形状变为 (1000, 28, 28)
# # 指定存储路径
# save_dir = f'E:/Gen_data/vgg_pil/mnist/{value_to_find}'
#
# # 如果目录不存在，则创建它
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
#
#
# for i in range(num_images):
#     fig = plt.figure(figsize=(28 / 28, 28 / 28), dpi=28)  # 创建特定大小的画布
#     plt.imshow(np.abs(generated_X_grayscale_np[i]), cmap='gray')  # 灰度图像
#     plt.axis('off')  # 关闭坐标轴
#     output_path = os.path.join(save_dir, f'Sampleimage_{i}.png')
#     plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 去除所有边距
#     plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
#     plt.close()
#
# print(f"所有图像已保存到 {save_dir} 文件夹中。")
