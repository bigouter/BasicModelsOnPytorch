import torch
from torch import nn
from torchsummary import summary

class LeNet(nn.Module):
    '''LeNet网络结构'''
    def __init__(self,input_dim,output_dim):
        super(LeNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_dim,out_channels=6,kernel_size=5,stride=1)     # 输出 28x28x6
        self.sigmoid1 = nn.Sigmoid()
        self.mp1 = nn.MaxPool2d(kernel_size=2,stride=2)                       # 输出 14x14x6
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1)   #输出 10x10x16
        self.relu2 = nn.ReLU()
        self.mp2 = nn.MaxPool2d(kernel_size=2,stride=2)                       # 输出 5x5x16
        self.conv3 = nn.Conv2d(in_channels=16,out_channels=120,kernel_size=5,stride=1)   # 输出 1x1x120
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(120,84)
        self.relu4 = nn.ReLU()
        self.linear2 = nn.Linear(84,output_dim)
        self.init_weights()

    def forward(self,X):
        X = self.conv1(X)
        X = self.sigmoid1(X)
        X = self.mp1(X)
        X = self.conv2(X)
        X = self.relu2(X)
        X = self.mp2(X)
        X = self.conv3(X)
        X = self.relu3(X)
        X = self.flatten(X)
        X = self.linear1(X)
        X = self.relu4(X)
        X = self.linear2(X)
        return X
    
    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer,(nn.Conv2d,nn.Linear)):
                torch.nn.init.xavier_normal_(layer.weight)
                torch.nn.init.constant_(layer.bias,0.1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # 查看模型结构
    model = LeNet(1,10).to(device = device)
    print(summary(model,(1,32,32)))
























