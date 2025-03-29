import torch
import torch.nn as nn
from torchsummary import summary

class VGG19(nn.Module):
    '''VGG19网络结构'''
    def __init__(self,input_dim,output_dim):
        super(VGG19,self).__init__()

        self.conv_1 = nn.Conv2d(input_dim,64,kernel_size = 3,stride = 1,padding = 1)
        self.relu_2 = nn.ReLU()                                                     # 输出 224x224x64
    
        self.conv_3 = nn.Conv2d(64,64,kernel_size = 3,stride = 1,padding = 1)
        self.relu_4 = nn.ReLU()                                                     # 输出 224x224x64
        
        self.maxpool_5 = nn.MaxPool2d(kernel_size = 2,stride = 2)                   # 输出112x112x64

        self.conv_6 = nn.Conv2d(64,128,kernel_size = 3,stride = 1,padding = 1)  
        self.relu_7 = nn.ReLU()                                                     # 输出 112x112x128

        self.conv_8 = nn.Conv2d(128,256,kernel_size = 3,stride = 1,padding = 1)
        self.relu_9 = nn.ReLU()                                                     # 输出 112x112x256

        self.maxpool_10 = nn.MaxPool2d(kernel_size = 2,stride = 2)                  # 输出56x56x256

        self.conv_11 = nn.Conv2d(256,256,kernel_size = 3,stride = 1,padding = 1)
        self.relu_12 = nn.ReLU()                                                    # 输出56x56x256
        self.conv_13 = nn.Conv2d(256,256,kernel_size = 3,stride = 1,padding = 1)
        self.relu_14 = nn.ReLU()                                                    # 输出56x56x256
        self.conv_15 = nn.Conv2d(256,256,kernel_size = 3,stride = 1,padding = 1)
        self.relu_16 = nn.ReLU()                                                    # 输出 56x56x256
        self.conv_17 = nn.Conv2d(256,512,kernel_size = 3,stride = 1,padding = 1)
        self.relu_18 = nn.ReLU()                                                    # 输出 56x56x512

        self.maxpool_19 = nn.MaxPool2d(kernel_size = 2,stride = 2)                  # 输出 28x28x512

        self.conv_20 = nn.Conv2d(512,512,kernel_size = 3,stride = 1,padding = 1)
        self.relu_21 = nn.ReLU()                                                    # 输出 28x28x512
        self.conv_22 = nn.Conv2d(512,512,kernel_size = 3,stride = 1,padding = 1)
        self.relu_23 = nn.ReLU()                                                    # 输出28x28x512
        self.conv_24 = nn.Conv2d(512,512,kernel_size = 3,stride = 1,padding = 1)
        self.relu_25 = nn.ReLU()                                                    # 输出28x28x512
        self.conv_26 = nn.Conv2d(512,512,kernel_size = 3,stride = 1,padding = 1)
        self.relu_27 = nn.ReLU()                                                    # 输出28x28x512

        self.maxpool_28 = nn.MaxPool2d(kernel_size = 2,stride = 2)                  # 输出14x14x512
        
        self.conv_29 = nn.Conv2d(512,512,kernel_size = 3,stride = 1,padding = 1)
        self.relu_30 = nn.ReLU()                                                    # 输出14x14x512
        self.conv_31 = nn.Conv2d(512,512,kernel_size = 3,stride = 1,padding = 1)
        self.relu_32 = nn.ReLU()                                                    # 输出14x14x512
        self.conv_33 = nn.Conv2d(512,512,kernel_size = 3,stride = 1,padding = 1)
        self.relu_34 = nn.ReLU()                                                    # 输出14x14x512
        self.conv_35 = nn.Conv2d(512,512,kernel_size = 3,stride = 1,padding = 1)
        self.relu_36 = nn.ReLU()                                                    # 输出14x14x512

        self.maxpool_37 = nn.MaxPool2d(kernel_size = 2,stride = 2)                  # 输出7x7x512

        self.flatten_38 = nn.Flatten()
        self.linear_39 = nn.Linear(7*7*512,4096)
        self.relu_40 = nn.ReLU()
        self.linear_41 = nn.Linear(4096,4096)
        self.relu_42 = nn.ReLU()
        self.linear_43 = nn.Linear(4096,output_dim)

        self.init_params()

    def forward(self,x):
        x = self.conv_1(x)
        x = self.relu_2(x)
        x = self.conv_3(x)
        x = self.relu_4(x)
        x = self.maxpool_5(x)
        x = self.conv_6(x)
        x = self.relu_7(x)
        x = self.conv_8(x)
        x = self.relu_9(x)
        x = self.maxpool_10(x)
        x = self.conv_11(x)
        x = self.relu_12(x)
        x = self.conv_13(x)
        x = self.relu_14(x)
        x = self.conv_15(x)
        x = self.relu_16(x)
        x = self.conv_17(x)
        x = self.relu_18(x)
        x = self.maxpool_19(x)
        x = self.conv_20(x)
        x = self.relu_21(x)
        x = self.conv_22(x)
        x = self.relu_23(x)
        x = self.conv_24(x)
        x = self.relu_25(x)
        x = self.conv_26(x)
        x = self.relu_27(x)
        x = self.maxpool_28(x)
        x = self.conv_29(x)
        x = self.relu_30(x)
        x = self.conv_31(x)
        x = self.relu_32(x)
        x = self.conv_33(x)
        x = self.relu_34(x)
        x = self.conv_35(x)
        x = self.relu_36(x)
        x = self.maxpool_37(x)
        x = self.flatten_38(x)
        x = self.linear_39(x)
        x = self.relu_40(x)
        x = self.linear_41(x)
        x = self.relu_42(x)
        x = self.linear_43(x)
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
    model = VGG19(3,10).to(device = device)
    print(summary(model,(3,224,224)))


