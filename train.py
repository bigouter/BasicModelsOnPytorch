import os
import sys
import time
import copy
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from model.LeNet import LeNet
import matplotlib.pyplot as plt
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST




def train_val_process(Data_save_path,Batch_size):
    train_data = FashionMNIST(root = Data_save_path,
                              train = True,
                              transform = transforms.Compose([transforms.Resize(size = 32),transforms.ToTensor()]),
                              download = True)

    train_data,val_data = Data.random_split(train_data,[round(0.8 * len(train_data)),round(0.2 * len(train_data))])

    train_dataloader = Data.DataLoader(dataset = train_data,
                                       batch_size = 128,
                                       shuffle = True,
                                       num_workers = 8)
    
    val_dataloader = Data.DataLoader(dataset = val_data,
                                     batch_size = Batch_size,
                                     shuffle = True,
                                     num_workers = 8)
    
    return train_dataloader,val_dataloader


def train_model_process(model,train_dataloader,val_dataloader,num_epochs,lr_rate = 0.001,best_weights_pth = None):
    # 设定训练所用道德设备，有GPU用GPU，没有GPU用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 使用Adam优化器，学习率为0。001
    optimizer = torch.optim.Adam(model.parameters(),lr = lr_rate)
    # 损失函数苏为交叉熵函数(内置softmax函数)
    criterion = nn.CrossEntropyLoss()
    # 将模型放入训练设备中
    model = model.to(device)

    ##############################################################################################################################
    # 复制模型参数
    best_model_wts = copy.deepcopy(model.state_dict())
    ##############################################################################################################################

    # 初始化参数
    # 最高准确度
    best_acc = 0.0
    # 训练集损失列表
    train_loss_all = []
    # 训练集准确度列表
    train_acc_all = []
    # 验证集损失列表
    val_loss_all = []
    # 验证集准确度列表
    val_acc_all = []
  
    for epoch in range(num_epochs):
        # 当前时间
        since = time.time()

        print("-" * 40 + f"Epoch {epoch + 1}/{num_epochs} starts" + "-" * 40)

        # 初始化参数
        # 训练集损失函数
        train_loss = 0.0
        # 训练集准确度
        train_corrects = 0
        # 训练样本数量
        train_num = 0

        # 验证集损失函数
        val_loss = 0.0
        # 验证集准确度
        val_corrects = 0
        # 验证样本数量
        val_num = 0

        # 训练
        for train_step,(images,labels) in enumerate(train_dataloader):
            # 将特征放入训练设备中
            images = images.to(device)
            labels = labels.to(device)

            model.train()

            # 输出的output是一组tensor,形状为（batch_size,output_dim）
            output = model(images)
            # 得到同组值最大的下标
            pre_lab = torch.argmax(output,dim = 1)
            # 计算损失
            loss = criterion(output,labels)
            # 优化算法梯度清零
            optimizer.zero_grad()
            # 损失函数反向传播
            loss.backward()
            # 参数更新
            optimizer.step()

            # 得到总的损失
            train_loss += loss.item() * images.size(0)
            # 预测正确的数量
            train_corrects += torch.sum(pre_lab == labels.data)
            # 加上训练的样本数量
            train_num += images.size(0)

        # 验证（验证时不需要反向传播）
        for val_step,(images,labels) in enumerate(val_dataloader):
            # 将数据放入设备中
            images,labels = images.to(device),labels.to(device)

            model.eval()

            output = model(images)
            # torch.argmax()包含了softmax()
            pre_lab = torch.argmax(output,dim = 1)

            loss = criterion(output,labels)

            val_loss += loss.item() * images.size(0)
            val_corrects += torch.sum(pre_lab == labels.data)
            val_num += images.size(0)

        time_cost = time.time() - since
        # （训练）计算并保存每轮迭代的loss值和准确率
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)  

        # （验证）计算并保存每轮迭代的loss值和准确率
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)
        
        print(f"Epoch {epoch + 1}, Train Loss:{train_loss_all[-1]:.4f},Train Acc:{train_acc_all[-1]:.4f}")
        print(f"Epoch {epoch + 1}, val Loss:{val_loss_all[-1]:.4f},val Acc:{val_acc_all[-1]:.4f}")
        print(f"本轮训练耗时{time_cost:.3f} sec")

        # 保存最优参数
        if val_acc_all[-1] > best_acc:
            # 保存当前的最高准确度
            best_acc = val_acc_all[-1]
            # 深拷贝保存当前的最优权重
            best_model_wts = model.state_dict()

        # 保存及选择最优参数
        torch.save(best_model_wts,best_weights_pth)

    # 绘图用到的数据
    return num_epochs,train_loss_all,val_loss_all,train_acc_all,val_acc_all


def draw_loss_acc_curve(save_path,num_epochs,train_loss_all,val_loss_all,train_acc_all,val_acc_all):
    '''绘图（train_loss、train_acc、val_loss、val_acc四条曲线画到一张图上）'''

    epochs = list(range(num_epochs))
    train_loss_all = train_loss_all
    val_loss_all = val_loss_all
    train_acc_all = train_acc_all
    val_acc_all = val_acc_all
    
    plt.plot(epochs,train_loss_all,"r-",label = "train_loss")
    plt.plot(epochs,val_loss_all,"g-.",label = "val_loss")
    plt.plot(epochs,train_acc_all,"b:",label = "train_acc")
    plt.plot(epochs,val_acc_all,"y--",label = "val_acc")
    plt.xlabel("epoch")
    plt.ylabel("Performance Measures")
    plt.legend()
    # 再调用show之前保存图像
    plt.savefig(fname = f"{save_path}" + "acc_loss.jpg",dpi = 300,format = "jpg")
    plt.show()

if __name__ == "__main__":
    init_params={
        # 用到的模型
        "model":{
            "LeNet":True,
            "VGG":False,
            "OtherModel":False,
        },
        # 性能图片存储路径
        "performance_picture_save_path":None,
        # 最优权重文件保存路径
        "best_weights_pth":"./model/best_model_weights/",
        # 输入图像的通道数
        "input_dim":1,
        # 模型输出的维度
        "output_dim":10,
        # 训练轮次
        "epochs":10,
        # 数据集路径
        "dataset_path":"./datasets/data",
        # 批量大小
        "batch_size":128,
        # 学习率
        "lr_rate":0.001
    }

    if init_params["model"]["LeNet"]:
        model = LeNet(init_params["input_dim"],init_params["output_dim"])
        init_params["best_weights_pth"] += "LeNet_best_weights.pt"       
        init_params["performance_picture_save_path"] = "./pictures/"
        
    if init_params["performance_picture_save_path"] is None:
        print("请确认模型及其相关参数！\n")
        sys.exit(0)
    else:
        num_epochs,train_loss_all,val_loss_all,train_acc_all,val_acc_all = train_model_process(model,*train_val_process(init_params["dataset_path"],init_params["batch_size"]),init_params["epochs"],init_params["lr_rate"],init_params["best_weights_pth"])
        draw_loss_acc_curve(init_params["performance_picture_save_path"],num_epochs,train_loss_all,val_loss_all,train_acc_all,val_acc_all)
