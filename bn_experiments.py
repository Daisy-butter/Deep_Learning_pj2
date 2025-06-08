# bn_experiments.py
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import random
import numpy as np
import os

from model import VGG_A
from data_loader import load_cifar10

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ----------------------------
# 固定随机种子，保证可复现
# ----------------------------
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------------
# 训练 + 评估函数
# ----------------------------
def train_and_evaluate_vgg(model, train_loader, test_loader, folder_name, epochs=100, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses = []
    test_accuracies = []

    print(model)
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    start_time = time.time()  # 记录训练开始时间

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Evaluate on test set
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

        print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Test Acc={acc:.2f}%")

        scheduler.step()

    end_time = time.time()  # 记录训练结束时间
    total_time_sec = end_time - start_time
    max_acc = max(test_accuracies)

    print(f"Total Training Time: {total_time_sec:.2f} seconds")
    print(f"Highest Test Accuracy: {max_acc:.2f}%")

    # 保存模型
    model_path = os.path.join(folder_name, "model.pth")
    torch.save(model.state_dict(), model_path)

    # 保存训练曲线
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss 曲线
    axes[0].plot(train_losses, color='blue')
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)

    # Accuracy 曲线
    axes[1].plot(test_accuracies, color='green')
    axes[1].set_title('Test Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].grid(True)

    plt.tight_layout()
    curve_path = os.path.join(folder_name, "curve.png")
    plt.savefig(curve_path)
    plt.close()

    # 保存 summary.txt
    summary_path = os.path.join(folder_name, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Total Training Time: {total_time_sec:.2f} seconds\n")
        f.write(f"Highest Test Accuracy: {max_acc:.2f}%\n")

    print(f"Results saved to {folder_name}")

    return model, train_losses, test_accuracies

# ----------------------------
# 可视化BN对比图
# ----------------------------
def plot_bn_comparison(loss_bn, acc_bn, loss_no_bn, acc_no_bn, save_folder):
    plt.figure(figsize=(12, 5))

    # Loss 对比
    plt.subplot(1, 2, 1)
    plt.plot(loss_bn, label='With BN', color='blue')
    plt.plot(loss_no_bn, label='Without BN', color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.grid(True)

    # Accuracy 对比
    plt.subplot(1, 2, 2)
    plt.plot(acc_bn, label='With BN', color='blue')
    plt.plot(acc_no_bn, label='Without BN', color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Test Accuracy Comparison")
    plt.legend()
    plt.grid(True)

    plt.suptitle("Effect of Batch Normalization")
    plt.tight_layout()

    comparison_path = os.path.join(save_folder, "bn_comparison.png")
    plt.savefig(comparison_path)
    plt.close()

# ----------------------------
# 训练不同lr下的loss曲线
# ----------------------------
def train_losses_across_lr(model_fn, train_loader, lr_list, epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_losses = {}

    for lr in lr_list:
        print(f"\nTraining with lr={lr}")
        model = model_fn().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        train_losses = []

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            train_losses.append(avg_loss)
            scheduler.step()

            print(f"Epoch {epoch+1}/{epochs}, Loss={avg_loss:.4f}")

        all_losses[lr] = train_losses

    return all_losses

# ----------------------------
# 计算 max_curve 和 min_curve
# ----------------------------
def compute_max_min_curve(all_losses):
    n_epochs = len(next(iter(all_losses.values())))
    max_curve = []
    min_curve = []

    for epoch in range(n_epochs):
        losses_at_epoch = [losses[epoch] for losses in all_losses.values()]
        max_curve.append(max(losses_at_epoch))
        min_curve.append(min(losses_at_epoch))

    return max_curve, min_curve

# ----------------------------
# 绘制loss landscape图
# ----------------------------
def plot_loss_landscape(max_curve, min_curve, save_path, title="Loss Landscape"):
    plt.figure(figsize=(8, 6))
    epochs = range(1, len(max_curve) + 1)
    plt.fill_between(epochs, min_curve, max_curve, color='skyblue', alpha=0.5)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(title)
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# ----------------------------
# Main 主流程
# ----------------------------
if __name__ == "__main__":
    set_seed(42)

    # 创建 results_vgg 文件夹
    os.makedirs("results_vgg", exist_ok=True)

    # 加载数据
    train_loader, test_loader = load_cifar10()

    # 训练有BN的VGG-A
    print("\n=== Training VGG-A with BatchNorm ===")
    folder_bn = "results_vgg/vgg_bn"
    os.makedirs(folder_bn, exist_ok=True)
    model_bn = VGG_A(use_bn=True)
    model_bn, losses_bn, accs_bn = train_and_evaluate_vgg(model_bn, train_loader, test_loader, folder_bn, epochs=100)

    # 训练无BN的VGG-A
    print("\n=== Training VGG-A without BatchNorm ===")
    folder_no_bn = "results_vgg/vgg_no_bn"
    os.makedirs(folder_no_bn, exist_ok=True)
    model_no_bn = VGG_A(use_bn=False)
    model_no_bn, losses_no_bn, accs_no_bn = train_and_evaluate_vgg(model_no_bn, train_loader, test_loader, folder_no_bn, epochs=100)

    # 生成对比图，保存在 results_vgg 根目录
    plot_bn_comparison(losses_bn, accs_bn, losses_no_bn, accs_no_bn, save_folder="results_vgg")
    print("\nBN Comparison plot saved to 'results_vgg/bn_comparison.png'")

    # 2.3 Loss Landscape Experiment
    lr_list = [1e-3, 2e-3, 1e-4, 5e-4]

    # With BN
    print("\n=== Loss Landscape: VGG-A with BN ===")
    all_losses_bn = train_losses_across_lr(lambda: VGG_A(use_bn=True), train_loader, lr_list, epochs=100)
    max_curve_bn, min_curve_bn = compute_max_min_curve(all_losses_bn)
    plot_loss_landscape(max_curve_bn, min_curve_bn, save_path="results_vgg/loss_landscape_bn.png", title="Loss Landscape - VGG-A with BN")

    # Without BN
    print("\n=== Loss Landscape: VGG-A without BN ===")
    all_losses_no_bn = train_losses_across_lr(lambda: VGG_A(use_bn=False), train_loader, lr_list, epochs=100)
    max_curve_no_bn, min_curve_no_bn = compute_max_min_curve(all_losses_no_bn)
    plot_loss_landscape(max_curve_no_bn, min_curve_no_bn, save_path="results_vgg/loss_landscape_no_bn.png", title="Loss Landscape - VGG-A without BN")
