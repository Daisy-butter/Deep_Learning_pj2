# models.py
import torch.nn as nn
import torch.nn.functional as F


class BaseCNN(nn.Module):
    """
    基础卷积神经网络模型，包含以下组件：
    - 2D卷积层
    - 2D池化层
    - 全连接层
    - 激活函数（ReLU）
    - 可选组件：批标准化（BatchNorm）、Dropout
    """
    def __init__(self, use_bn=False, use_dropout=False):
        super().__init__()
        # 卷积层1: 输入通道3（RGB），输出通道64，卷积核3x3，填充1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64) if use_bn else nn.Identity()  # 批标准化层（可选）
        # 最大池化层，窗口大小2x2，步长2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 卷积层2: 输入通道64，输出通道128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128) if use_bn else nn.Identity()
        
        # 全连接层1: 输入维度128*8*8（经过两次池化后尺寸为8x8），输出512
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.dropout = nn.Dropout(0.5) if use_dropout else nn.Identity()  # Dropout（可选）
        # 全连接层2: 输出10类
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        # 卷积层1 -> BatchNorm -> ReLU -> 池化
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # 卷积层2 -> BatchNorm -> ReLU -> 池化
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # 展平特征图
        x = x.view(-1, 128 * 8 * 8)
        # 全连接层1 -> ReLU -> Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # 全连接层2（输出层）
        x = self.fc2(x)
        return x


class VGG_A(nn.Module):
    """
    VGG-A网络结构，支持带BN和不带BN的配置。
    输入尺寸：32x32x3（CIFAR-10）
    """
    def __init__(self, use_bn=True):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: Conv -> BN（可选） -> ReLU -> Pool
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2: Conv -> BN -> ReLU -> Pool
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3: Conv -> BN -> ReLU -> Pool
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 全连接层（修复括号缺失问题）
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),  # 输入尺寸根据池化层调整
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )  # 添加闭合括号
    
    def forward(self, x):
        # 特征提取
        x = self.features(x)
        # 展平
        x = x.view(x.size(0), -1)
        # 分类器
        x = self.classifier(x)
        return x