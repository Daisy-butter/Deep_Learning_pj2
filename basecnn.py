import os
import time  # 新增
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_loader import load_cifar10

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# === 模型定义（包含残差连接） ===
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

# === 可视化工具 ===
def visualize_kernels(model, save_path):
    weights = model.block1.conv1.weight.data.clone().cpu()
    grid = torchvision.utils.make_grid(weights, nrow=8, normalize=True, padding=1)
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.title("Conv1 Kernels")
    plt.imshow(grid.permute(1, 2, 0))
    plt.savefig(os.path.join(save_path, "conv1_kernels.png"))
    plt.close()

def visualize_feature_map(feature_map, save_path):
    fmap = feature_map[0]  # 取第一个样本
    fig, axs = plt.subplots(1, min(6, fmap.shape[0]), figsize=(12, 3))
    for i in range(min(6, fmap.shape[0])):
        axs[i].imshow(fmap[i].cpu(), cmap='viridis')
        axs[i].axis('off')
    plt.suptitle("Feature Maps")
    plt.savefig(os.path.join(save_path, "feature_maps.png"))
    plt.close()

# === 训练函数 ===
def train_and_evaluate(activation, loss_fn_name, filters):
    folder = f"results/{activation}_{loss_fn_name}_{filters[0]}-{filters[1]}"
    os.makedirs(folder, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedCNN(filters=filters, activation=activation).to(device)

    train_loader, test_loader = load_cifar10()
    criterion = nn.CrossEntropyLoss() if loss_fn_name == 'ce' else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)  # 修改epoch

    train_losses = []
    test_accuracies = []

    print(f"[{activation}-{loss_fn_name}-{filters}] Start Training")
    
    start_time = time.time()  # 记录开始时间

    for epoch in range(100):  # 修改epoch
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            if loss_fn_name == 'mse':
                labels_onehot = F.one_hot(labels, 10).float()
                loss = criterion(outputs, labels_onehot)
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = 100 * correct / total
        test_accuracies.append(acc)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={acc:.2f}%")
        scheduler.step()

    end_time = time.time()  # 记录结束时间
    total_time_sec = end_time - start_time
    max_acc = max(test_accuracies)
    print(f"Total Training Time: {total_time_sec:.2f} seconds")
    print(f"Highest Test Accuracy: {max_acc:.2f}%")

    torch.save(model.state_dict(), os.path.join(folder, "model.pth"))
    visualize_kernels(model, folder)

    sample_input, _ = next(iter(test_loader))
    sample_input = sample_input.to(device)
    with torch.no_grad():
        _ = model(sample_input)
    visualize_feature_map(model.feature_map, folder)

    # 保存训练曲线（改进：使用子图分别画 loss 和 accuracy）
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 绘制 Loss 曲线
    axes[0].plot(train_losses, color='blue')
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)

    # 绘制 Accuracy 曲线
    axes[1].plot(test_accuracies, color='green')
    axes[1].set_title('Test Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].grid(True)

    # 自动调整子图间距并保存
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "curve.png"))
    plt.close()

    # === 保存 summary.txt ===
    with open(os.path.join(folder, "summary.txt"), "w") as f:
        f.write(f"Total Training Time: {total_time_sec:.2f} seconds\n")
        f.write(f"Highest Test Accuracy: {max_acc:.2f}%\n")

    print(f"[{activation}-{loss_fn_name}-{filters}] Training Complete and results saved to {folder}")

# === 主执行函数 ===
if __name__ == "__main__":
    configs = [
        ('leaky_relu', 'ce', (64, 128)),
        ('sigmoid', 'ce', (64, 128)),
        ('leaky_relu', 'mse', (64, 128)),
        ('leaky_relu', 'ce', (32, 64)),
        ('sigmoid', 'ce', (32, 64)),
        ('leaky_relu', 'mse', (32, 64)),
    ]

    for activation, loss_fn, filters in configs:
        train_and_evaluate(activation, loss_fn, filters)
