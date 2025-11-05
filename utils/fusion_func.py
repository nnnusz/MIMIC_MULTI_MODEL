import torch
import torch.nn as nn
import numpy as np
import h5py
from torch.utils.data import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, \
    average_precision_score


# 1. 数据加载与预处理
class MultimodalDataset(Dataset):
    def __init__(self, image_features, text_features, labels):
        self.image_features = torch.tensor(image_features, dtype=torch.float32)
        self.text_features = torch.tensor(text_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "image": self.image_features[idx],
            "text": self.text_features[idx],
            "label": self.labels[idx]
        }


# 加载HDF5数据
def load_h5_data(image_path, text_path):
    with h5py.File(image_path, 'r') as f_img:
        image_features = f_img['features'][:]
        image_labels = f_img['labels'][:]

    with h5py.File(text_path, 'r') as f_txt:
        text_features = f_txt['features'][:]
        text_labels = f_txt['labels'][:]

    # 确保标签一致
    assert np.array_equal(image_labels, text_labels), "图像和文本标签不匹配"

    return image_features, text_features, image_labels


# 文本特征增强函数
def enhance_text_features(text_features, labels, enhancement_factor=0.3):
    """
    基于文本特征的强判别性进行增强
    :param text_features: 原始文本特征
    :param labels: 样本标签
    :param enhancement_factor: 增强因子 (0-1)
    :return: 增强后的文本特征
    """
    # 计算每个类别的中心
    unique_labels = np.unique(labels)
    class_centers = []
    for label in unique_labels:
        class_features = text_features[labels == label]
        class_center = np.mean(class_features, axis=0)
        class_centers.append(class_center)

    # 将每个样本特征向类中心方向移动
    enhanced_features = np.zeros_like(text_features)
    for i, (feat, label) in enumerate(zip(text_features, labels)):
        class_center = class_centers[label]
        direction = class_center - feat
        enhanced_features[i] = feat + enhancement_factor * direction

    return enhanced_features

# 3. 训练与评估函数
class GuidedLoss(nn.Module):
    """文本引导损失函数"""

    def __init__(self, alpha=0.7):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.alpha = alpha

    def forward(self, multimodal_output, text_output, labels):
        multimodal_loss = self.ce(multimodal_output, labels)
        text_loss = self.ce(text_output, labels)
        return self.alpha * text_loss + (1 - self.alpha) * multimodal_loss


def train_epoch(model, dataloader, optimizer, criterion, device, phase='train'):
    model.train() if phase == 'train' else model.eval()
    total_loss = 0.0
    multimodal_preds = []
    all_labels = []

    for batch in dataloader:
        images = batch['image'].to(device)
        texts = batch['text'].to(device)
        labels = batch['label'].to(device)

        if phase == 'train':
            optimizer.zero_grad()

        multimodal_logits, text_logits = model(images, texts)
        loss = criterion(multimodal_logits, text_logits, labels)

        if phase == 'train':
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        _, multimodal_pred = torch.max(multimodal_logits, 1)
        multimodal_preds.extend(multimodal_pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    multimodal_acc = accuracy_score(all_labels, multimodal_preds)

    return avg_loss, multimodal_acc


def calculate_metrics(true_labels, pred_labels):
    """
    计算分类模型的性能指标
    """
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)

    return accuracy, precision, recall, f1

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_multimodal_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            texts = batch['text'].to(device)
            labels = batch['label'].to(device)

            multimodal_logits, text_logits = model(images, texts)
            loss = criterion(multimodal_logits, text_logits, labels)

            total_loss += loss.item()
            _, multimodal_pred = torch.max(multimodal_logits, 1)
            all_multimodal_preds.extend(multimodal_pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    multimodal_acc = accuracy_score(all_labels, all_multimodal_preds)
    multimodal_precision = precision_score(all_labels, all_multimodal_preds, average='weighted', zero_division=0)
    multimodal_recall = recall_score(all_labels, all_multimodal_preds, average='weighted', zero_division=0)
    multimodal_f1 = f1_score(all_labels, all_multimodal_preds, average='weighted', zero_division=0)

    auc_score = roc_auc_score(all_labels, all_multimodal_preds)
    aupr_score = average_precision_score(all_labels, all_multimodal_preds)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_multimodal_preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        'loss': avg_loss,
        'accuracy': multimodal_acc,
        'precision': multimodal_precision,
        'recall': multimodal_recall,
        'f1': multimodal_f1,
        'preds': all_multimodal_preds,
        'labels': all_labels,
        'auc_score': auc_score,
        'aupr_score': aupr_score,
        'specificity': specificity
    }

def evaluate_detail(model, dataloader, criterion, device):
    """评估模型性能并收集注意力权重"""
    model.eval()
    total_loss = 0.0
    all_multimodal_preds = []
    all_labels = []
    all_probs = []  # 存储预测概率
    all_attention_weights = []  # 存储注意力权重

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            texts = batch['text'].to(device)
            labels = batch['label'].to(device)

            multimodal_logits, text_logits = model(images, texts)
            loss = criterion(multimodal_logits, text_logits, labels)

            total_loss += loss.item()
            #预测结果
            probs = torch.softmax(multimodal_logits, dim=1)
            _, multimodal_pred = torch.max(multimodal_logits, 1)
            all_multimodal_preds.extend(multimodal_pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # 获取注意力权重
            attention_weights = model.get_attention_weights()
            if attention_weights is not None:
                all_attention_weights.append(attention_weights.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    #计算评估指标
    multimodal_acc = accuracy_score(all_labels, all_multimodal_preds)
    multimodal_precision = precision_score(all_labels, all_multimodal_preds, average='weighted', zero_division=0)
    multimodal_recall = recall_score(all_labels, all_multimodal_preds, average='weighted', zero_division=0)
    multimodal_f1 = f1_score(all_labels, all_multimodal_preds, average='weighted', zero_division=0)

    auc_score = roc_auc_score(all_labels, all_multimodal_preds)
    aupr_score = average_precision_score(all_labels, all_multimodal_preds)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_multimodal_preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # 计算平均注意力权重
    avg_attention_weights = np.mean(np.concatenate(all_attention_weights, axis=0),
                                    axis=0) if all_attention_weights else None

    return {
        'loss': avg_loss,
        'accuracy': multimodal_acc,
        'precision': multimodal_precision,
        'recall': multimodal_recall,
        'f1': multimodal_f1,
        'preds': all_multimodal_preds,
        'labels': all_labels,
        'auc_score': auc_score,
        'aupr_score': aupr_score,
        'specificity': specificity,
        'probs': np.array(all_probs),
        'attention_weights': avg_attention_weights
    }

def plot_confusion_matrix(labels, preds, class_names, save_path):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                annot_kws={"size": 20})
    plt.title('Test Set Confusion Matrix', fontsize=20)
    plt.ylabel('True Label', fontsize=20)
    plt.xlabel('Predicted Label', fontsize=20)
    plt.savefig(save_path)
    plt.close()


def plot_training_curves(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()