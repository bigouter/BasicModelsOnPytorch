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

def test_data_process(batch_size):
    test_data = FashionMNIST(root = "./dataset/data",
                             train = False,
                             transform=transforms.Compose([transforms.Resize(32),transforms.ToTensor()]),
                             download=True)
    
    test_dataloader = Data.DataLoader(dataset=test_data,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=0)

    return test_dataloader 


def test(model,test_dataloader,best_weights_pth = None):
    if best_weights_pth is None:
        print("请添加最优权重文件路径！")
        sys.exit(0)
    else:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        state_dict = torch.load(best_weights_pth)
        model.load_state_dict(state_dict)

        model.eval()

        test_corrects = 0
        test_num = 0

        print("-"*30 + "测试开始" + "-"*30 + "\n")
        for image,label in test_dataloader:
            
            image = image.to(device)
            label = label.to(device)
            
            output = model(image)

            pre_lab = torch.argmax(output,dim = 1)

            test_corrects += (pre_lab == label.data).sum()
            test_num += image.size(0)

        print(" " * 25 + f"测试精度：{(test_corrects / test_num):.3f}")
        print("\n" + "-"*30 + "测试结束" + "-"*30)



if __name__ == "__main__":
    init_params = {
        "best_weights_path":None,
        "input_dim":1,
        "output_dim":10,   
        "batch_size":1,     
        "model":{
            "LeNet":True,
            "OtherModel":False
        }
    }

    if init_params["model"]["LeNet"] is True:
        model = LeNet(init_params["input_dim"],init_params["output_dim"])
        init_params["best_weights_path"] = "./model/best_model_weights/LeNet_best_weights.pt"
  
    test_data_loader = test_data_process(init_params["batch_size"])
    test(model,test_data_loader,init_params["best_weights_path"])

