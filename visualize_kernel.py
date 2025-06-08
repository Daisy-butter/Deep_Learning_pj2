import os
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# === 模型定义（跟 train 时保持一致）===
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation_fn, use_bn=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.activation = activation_fn
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.activation(out + identity)

class ImprovedCNN(nn.Module):
    def __init__(self, filters=(64, 128), activation='leaky_relu'):
        super().__init__()
        act_fn = nn.LeakyReLU() if activation == 'leaky_relu' else nn.Sigmoid()
        self.block1 = ResidualBlock(3, filters[0], act_fn)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.block2 = ResidualBlock(filters[0], filters[1], act_fn)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(filters[1]*8*8, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
        self.activation = act_fn

    def forward(self, x):
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        self.feature_map = x.detach()  # 保存中间特征图
        x = x.view(x.size(0), -1)
        x = self.dropout(self.activation(self.fc1(x)))
        return self.fc2(x)

# === 可视化函数 ===
def visualize_conv_weights(conv_layer, save_path, title):
    weights = conv_layer.weight.data.clone().cpu()  # shape: (out_channels, in_channels, kH, kW)
    out_channels, in_channels, kH, kW = weights.shape

    # 取每个 out_channel 对 in_channel=0 的 kernel
    kernels = weights[:, 0, :, :]  # shape: (out_channels, kH, kW)

    ncols = 8
    nrows = (out_channels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5))
    axes = axes.flatten()

    for i in range(out_channels):
        axes[i].imshow(kernels[i], cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Filter {i}')

    # 把多余的子图隐藏
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# === 主执行流程 ===
if __name__ == "__main__":
    configs = [
        ('leaky_relu', 'ce', (64, 128)),
        ('sigmoid', 'ce', (64, 128)),
        ('leaky_relu', 'mse', (64, 128)),
        ('leaky_relu', 'ce', (32, 64)),
        ('sigmoid', 'ce', (32, 64)),
        ('leaky_relu', 'mse', (32, 64)),
    ]

    for activation, loss_fn_name, filters in configs:
        folder = f"results/{activation}_{loss_fn_name}_{filters[0]}-{filters[1]}"
        model_path = os.path.join(folder, "model.pth")

        # 初始化模型并加载权重
        model = ImprovedCNN(filters=filters, activation=activation)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()

        # 可视化 block2.conv1 卷积核
        conv1_save_path = os.path.join(folder, "block2_conv1_kernels.png")
        visualize_conv_weights(model.block2.conv1, conv1_save_path, "Block2 Conv1 Kernels")

        # 可视化 block2.conv2 卷积核
        conv2_save_path = os.path.join(folder, "block2_conv2_kernels.png")
        visualize_conv_weights(model.block2.conv2, conv2_save_path, "Block2 Conv2 Kernels")

        print(f"[{activation}-{loss_fn_name}-{filters}] 卷积核可视化已保存到 {folder}/")
