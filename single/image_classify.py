import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn.functional as F
import torchxrayvision as xrv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
import time

#加载数据集
DATA_TYPE = 'mimic'
from mimic_data_loader import get_data

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#训练参数
BATCH_SIZE = 64
EPOCH = 20
#模型选择
MODEL_NAME = 'resnet'

# 数据预处理和增强
if MODEL_NAME == 'densenet':
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.Grayscale(num_output_channels=1),  # 确保单通道输出
            transforms.ToTensor(),
            # 使用torchxrayvision推荐的标准化参数
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),  # 确保单通道输出
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    }
else:
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

# 自定义数据集类
class ChestXrayDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            if MODEL_NAME == 'densenet':
                image = Image.open(img_path).convert('L')  # 单通道
            else:
                image = Image.open(img_path).convert('RGB')  # 确保三通道
            label = self.labels[idx]

            if self.transform:
                image = self.transform(image)

            return image, torch.tensor(label, dtype=torch.float32)
        except OSError as e:
            print(e,img_path)


class CustomDenseNet(xrv.models.DenseNet):
    def __init__(self, weights="densenet121-res224-chex", apply_sigmoid=False, op_threshs=None):
        super().__init__(weights=weights, apply_sigmoid=apply_sigmoid, op_threshs=op_threshs)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return self.classifier(out)

# 训练函数
def train_model(model, train_loader, val_loader, train_dataset, val_dataset, criterion, optimizer, num_epochs=20):
    best_f1 = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)
        start_t = time.time()

        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device).view(-1, 1)  # [batch, 1]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # 关键修改：添加squeeze(1)确保形状匹配
            preds = (outputs > 0.5).float().squeeze(1)  # [batch]
            labels_flat = labels.squeeze(1)  # [batch]
            running_corrects += torch.sum(preds == labels_flat)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.cpu().numpy())

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).view(-1, 1)  # [batch, 1]

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)

                # 关键修改：添加squeeze(1)确保形状匹配
                preds = (outputs > 0.5).float().squeeze(1)  # [batch]
                labels_flat = labels.squeeze(1)  # [batch]
                val_running_corrects += torch.sum(preds == labels_flat)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels_flat.cpu().numpy())

        val_epoch_loss = val_running_loss / len(val_dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_dataset)
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc.cpu().numpy())

        # 计算其他指标
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)

        print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')
        print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

        # 保存最佳模型
        if f1 > best_f1:
            best_f1 = f1
            #torch.save(model.state_dict(), f'{DATA_TYPE}_{MODEL_NAME}_model.pth')  # 只保存state_dict
            print('Saved best model with F1: {:.4f}'.format(best_f1))

        end_t = time.time()
        print('一轮训练时长：', end_t - start_t)

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.legend()
    plt.title('Accuracy')
    plt.savefig('training_curve.png')
    plt.close()

    return model


# 最终评估函数
def evaluate_model(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).view(-1, 1)  # [batch, 1]

            outputs = model(inputs)

            # 关键修改：添加squeeze(1)确保形状匹配
            preds = (outputs > 0.5).float().squeeze(1)  # [batch]
            labels_flat = labels.squeeze(1)  # [batch]

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_flat.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print('\nFinal Evaluation:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print('Confusion Matrix:')
    print(cm)

    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    classes = ['Normal', 'Pneumonia']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

    return accuracy, precision, recall, f1

def main():
    images,reports,labels = get_data()

    X_train, X_test, y_train, y_test = train_test_split(
        images, labels,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    # 创建数据集
    train_dataset = ChestXrayDataset(X_train, y_train, transform=data_transforms['train'])
    val_dataset = ChestXrayDataset(X_test, y_test, transform=data_transforms['val'])

    # 计算类别权重以处理不平衡问题
    class_counts = np.bincount(y_train)
    class_weights = 1. / class_counts
    weights = class_weights[y_train]
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))

    # 创建数据加载器
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if MODEL_NAME == 'densenet':
        # 加载densenet121
        model = CustomDenseNet(
            weights="densenet121-res224-chex",
            apply_sigmoid=False,
            op_threshs=None
        )
        # 修改分类器
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    else:
        # 加载预训练的ResNet50模型
        model = models.resnet50(pretrained=True)
        # 修改最后一层全连接层用于二分类
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid()
        )

    # 冻结模型的所有参数（可选，微调时通常解冻部分层）
    # for param in model.parameters():
    #     param.requires_grad = False

    model = model.to(device)

    # 定义损失函数和优化器
    # 使用带权重的BCELoss处理类别不平衡
    pos_weight = torch.tensor([class_counts[0] / class_counts[1]]).to(device)  # 正样本权重
    criterion = nn.BCELoss(weight=pos_weight)  # 也可以使用BCEWithLogitsLoss
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 训练模型
    trained_model = train_model(
        model, train_loader, val_loader,
        train_dataset, val_dataset,
        criterion, optimizer, EPOCH)

    # 加载最佳模型进行评估
    state_dict = torch.load(f'{DATA_TYPE}_{MODEL_NAME}_model.pth')
    if MODEL_NAME == 'densenet':
        best_model = CustomDenseNet(
            weights=None,  # 不加载预训练权重
            apply_sigmoid=False,
            op_threshs=None
        )
        best_model.classifier = nn.Sequential(
            nn.Linear(best_model.classifier.in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # 移除模型中不存在的键
        model_state_dict = best_model.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}

        # 加载过滤后的状态字典
        best_model.load_state_dict(filtered_state_dict)
    else:
        best_model = models.resnet50(pretrained=True)
        # 修改最后一层全连接层用于二分类
        num_ftrs = best_model.fc.in_features
        best_model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid()
        )
        best_model.load_state_dict(state_dict)
    best_model = best_model.to(device)

    # 在验证集上评估
    evaluate_model(best_model, val_loader)

if __name__ == '__main__':
    main()