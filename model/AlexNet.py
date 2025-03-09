import torch
import torch.nn as nn
from torchsummary import summary

class AlexNet(nn.Module):
    '''AlexNet网络结构'''
    def __init__(self,input_dim,output_dim):
        super(AlexNet,self).__init__()
        self.conv_1 = nn.Conv2d(in_channels = input_dim,out_channels = 96,kernel_size = 11,padding = 1,stride = 4)    # 输出 54x54x96
        self.relu_2 = nn.ReLU()
        self.pool_3 = nn.MaxPool2d(kernel_size = 3,stride = 2)                                                        # 输出 26x26x96
        self.conv_4 = nn.Conv2d(96,256,kernel_size=5,padding = 2,stride = 1)                                          # 
        self.relu_5 = nn.ReLU()
        self.pool_6 = nn.MaxPool2d(kernel_size = 3,stride = 2)
        self.conv_7 = nn.Conv2d(256,384,kernel_size = 3,padding = 1,stride = 1)
        self.relu_8 = nn.ReLU()
        self.conv_9 = nn.Conv2d(384,384,kernel_size = 3,padding = 1,stride = 1)
        self.relu_10 = nn.ReLU()
        self.conv_11 = nn.Conv2d(384,256,kernel_size = 3,padding = 1,stride = 1)
        self.relu_12 = nn.ReLU()
        self.pool_13 = nn.MaxPool2d(kernel_size = 3,stride = 2)
        self.flatten_14 = nn.Flatten()
        self.dense_15 = nn.Linear(256 * 5 * 5,4096)
        self.relu_16 = nn.ReLU()
        self.dropout_17 = nn.Dropout(p = 0.5)
        self.dense_18 = nn.Linear(4096,4096)
        self.relu_19 = nn.ReLU()
        self.dropout_20 = nn.Dropout(p = 0.5)
        self.dense_21 = nn.Linear(4096,output_dim)
        self.init_params()

    def forward(self,x):
        x = self.conv_1(x)
        x = self.relu_2(x)
        x = self.pool_3(x)
        x = self.conv_4(x)
        x = self.relu_5(x)
        x = self.pool_6(x)
        x = self.conv_7(x)
        x = self.relu_8(x)
        x = self.conv_9(x)
        x = self.relu_10(x)
        x = self.conv_11(x)
        x = self.relu_12(x)
        x = self.pool_13(x)
        x = self.flatten_14(x)
        x = self.dense_15(x)
        x = self.relu_16(x)
        x = self.dropout_17(x)
        x = self.dense_18(x)
        x = self.relu_19(x)
        x = self.dropout_20(x)
        x = self.dense_21(x)
        return x

    def init_params(self):
        for layer in self.modules():
            if isinstance(layer,(nn.Conv2d,nn.Linear)):
                torch.nn.init.xavier_normal_(layer.weight)
                torch.nn.init.constant_(layer.bias,0.1)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # 查看模型结构
    model = AlexNet(3,10).to(device = device)
    print(summary(model,(3,224,224)))



























