from torchvision.datasets import FashionMNIST
from torchvision import transforms
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt

train_data = FashionMNIST(root = "./data/",
                          train=True,
                          transform=transforms.Compose([transforms.Resize(224),transforms.ToTensor()]),
                          download=True)

train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=64,
                               shuffle=True,
                               num_workers=0)


# 训练集的标签
class_label = train_data.classes
print(class_label)

# 获得一个批次的数据
for step,(b_x,b_y) in enumerate(train_loader):
    if step > 0:
        break

# 将四维张量移除第一维，转换成numpy数组
batch_x = b_x.squeeze().numpy()
# 将张量转换成numpu数组
batch_y = b_y.squeeze().numpy()
print(batch_x.shape)
print(batch_y.shape)


# 可视化一个batch的图像
plt.figure(figsize=(12,5))
for ii in np.arange(len(batch_y)):
    plt.subplot(4,16,ii + 1)
    plt.imshow(batch_x[ii,:,:],cmap=plt.cm.gray)
    plt.title(class_label[batch_y[ii]],size = 10)
    plt.axis("off")
    plt.subplots_adjust(wspace=0.05)
plt.show()














