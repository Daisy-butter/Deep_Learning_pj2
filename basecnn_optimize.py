import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_loader import load_cifar10
from basecnn import ImprovedCNN, visualize_kernels, visualize_feature_map

def train_with_optimizer(opt_name, model, train_loader, test_loader, save_path, use_scheduler=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    if opt_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif opt_name == "Adam+StepLR":
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    scheduler = None
    if use_scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    train_losses, test_accuracies = [], []

    for epoch in range(50):  # 训练轮数适中
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"[{opt_name}] Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Evaluation
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

        print(f"[{opt_name}] Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={acc:.2f}%")
        if scheduler:
            scheduler.step()

    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "model.pth"))
    visualize_kernels(model, save_path)

    sample_input, _ = next(iter(test_loader))
    sample_input = sample_input.to(device)
    with torch.no_grad():
        _ = model(sample_input)
    visualize_feature_map(model.feature_map, save_path)

    # # 保存训练曲线
    # plt.plot(train_losses, label='Loss')
    # plt.plot(test_accuracies, label='Accuracy')
    # plt.legend()
    # plt.title(f"{opt_name} Training Curve")
    # plt.savefig(os.path.join(save_path, "curve.png"))
    # plt.close()

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
    plt.savefig(os.path.join(save_path, "curve.png"))
    plt.close()
    print(f"{opt_name} Training Complete and results saved to {save_path}")

if __name__ == "__main__":
    train_loader, test_loader = load_cifar10()
    optimizer_list = ["SGD", "Adam", "Adam+StepLR"]

    for opt_name in optimizer_list:
        print(f"\n===== Training with {opt_name} =====")
        model = ImprovedCNN(filters=(64, 128), activation='leaky_relu')
        save_dir = f"results_optimizers/{opt_name.replace('+', '_')}"
        use_scheduler = (opt_name == "Adam+StepLR")
        train_with_optimizer(opt_name, model, train_loader, test_loader, save_dir, use_scheduler)
