import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import random
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
import warnings
from tqdm import tqdm

# 忽略警告
warnings.filterwarnings('ignore')

# 加载不同数据源
DATA_TYPE = 'mimic'
from mimic_data_loader import get_data

# 设置随机种子确保可重复性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


set_seed()

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 自定义数据集类 - 修改为处理findings和impression分开
class MedicalReportDataset(Dataset):
    def __init__(self, reports, labels, tokenizer, max_len=256):
        self.reports = reports
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

        # 分离findings和impression
        self.findings = []
        self.impressions = []
        for report in reports:

            parts = report.split('####')
            findings_part = parts[0].strip() if len(parts) > 0 else ""
            impression_part = parts[1].strip() if len(parts) > 1 else ""

            self.findings.append(findings_part)
            self.impressions.append(impression_part)

    def __len__(self):
        return len(self.reports)

    def __getitem__(self, idx):
        findings = str(self.findings[idx])
        impression = str(self.impressions[idx])
        label = self.labels[idx]

        # 分别编码findings和impression
        findings_encoding = self.tokenizer.encode_plus(
            findings,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        impression_encoding = self.tokenizer.encode_plus(
            impression,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'findings_input_ids': findings_encoding['input_ids'].flatten(),
            'findings_attention_mask': findings_encoding['attention_mask'].flatten(),
            'impression_input_ids': impression_encoding['input_ids'].flatten(),
            'impression_attention_mask': impression_encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# 自定义模型 - 增加对findings和impression的加权处理
class WeightedMedicalBERT(nn.Module):
    def __init__(self, base_model, findings_weight=0.7, impression_weight=0.3):
        super(WeightedMedicalBERT, self).__init__()
        self.bert = base_model
        self.findings_weight = findings_weight
        self.impression_weight = impression_weight

    def forward(self, findings_input_ids, findings_attention_mask,
                impression_input_ids, impression_attention_mask, labels=None):
        # 处理findings部分
        findings_outputs = self.bert(
            input_ids=findings_input_ids,
            attention_mask=findings_attention_mask,
            labels=labels
        )

        # 处理impression部分
        impression_outputs = self.bert(
            input_ids=impression_input_ids,
            attention_mask=impression_attention_mask,
            labels=labels
        )

        # 加权融合
        weighted_logits = self.findings_weight * findings_outputs.logits + \
                          self.impression_weight * impression_outputs.logits

        # 计算损失
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(weighted_logits.view(-1, self.bert.config.num_labels), labels.view(-1))

        return {
            'loss': loss,
            'logits': weighted_logits
        }


# 训练函数 - 使用加权损失
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs):
    best_f1 = 0.0
    train_losses = []
    val_f1_scores = []

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        # 训练阶段
        model.train()
        epoch_train_loss = 0
        total_train_samples = 0

        for batch in tqdm(train_loader):
            findings_input_ids = batch['findings_input_ids'].to(device)
            findings_attention_mask = batch['findings_attention_mask'].to(device)
            impression_input_ids = batch['impression_input_ids'].to(device)
            impression_attention_mask = batch['impression_attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(
                findings_input_ids=findings_input_ids,
                findings_attention_mask=findings_attention_mask,
                impression_input_ids=impression_input_ids,
                impression_attention_mask=impression_attention_mask,
                labels=labels
            )

            loss = outputs['loss']
            epoch_train_loss += loss.item() * findings_input_ids.size(0)
            total_train_samples += findings_input_ids.size(0)

            loss.backward()
            optimizer.step()

        avg_train_loss = epoch_train_loss / total_train_samples
        train_losses.append(avg_train_loss)

        # 验证阶段
        model.eval()
        val_predictions = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                findings_input_ids = batch['findings_input_ids'].to(device)
                findings_attention_mask = batch['findings_attention_mask'].to(device)
                impression_input_ids = batch['impression_input_ids'].to(device)
                impression_attention_mask = batch['impression_attention_mask'].to(device)
                batch_labels = batch['labels'].cpu().numpy()

                outputs = model(
                    findings_input_ids=findings_input_ids,
                    findings_attention_mask=findings_attention_mask,
                    impression_input_ids=impression_input_ids,
                    impression_attention_mask=impression_attention_mask
                )

                _, preds = torch.max(outputs['logits'], dim=1)

                val_predictions.extend(preds.cpu().numpy())
                val_labels.extend(batch_labels)

        # 计算验证指标
        val_accuracy = accuracy_score(val_labels, val_predictions)
        val_precision = precision_score(val_labels, val_predictions)
        val_recall = recall_score(val_labels, val_predictions)
        val_f1 = f1_score(val_labels, val_predictions)
        val_f1_scores.append(val_f1)

        print(f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Acc: {val_accuracy:.4f}, Val Prec: {val_precision:.4f}, '
              f'Val Rec: {val_recall:.4f}, Val F1: {val_f1:.4f}')

        # 保存最佳模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_path = f'{DATA_TYPE}_text_model.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
                'f1': val_f1,
                'precision': val_precision,
                'recall': val_recall
            }, best_model_path)
            print(f"New best model saved with F1: {val_f1:.4f}")

    # 绘制训练损失和验证F1曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_f1_scores, label='Validation F1 Score', color='orange')
    plt.title('Validation F1 Score over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig('text_training_metrics.png')

    return model


# 评估函数
def evaluate_model(model, data_loader):
    checkpoint = torch.load(f'{DATA_TYPE}_text_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    predictions = []
    actual_labels = []

    with torch.no_grad():
        for batch in data_loader:
            findings_input_ids = batch['findings_input_ids'].to(device)
            findings_attention_mask = batch['findings_attention_mask'].to(device)
            impression_input_ids = batch['impression_input_ids'].to(device)
            impression_attention_mask = batch['impression_attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()

            outputs = model(
                findings_input_ids=findings_input_ids,
                findings_attention_mask=findings_attention_mask,
                impression_input_ids=impression_input_ids,
                impression_attention_mask=impression_attention_mask
            )

            _, preds = torch.max(outputs['logits'], dim=1)

            predictions.extend(preds.cpu().numpy())
            actual_labels.extend(labels)

    return actual_labels, predictions


# 主函数
def main(reports, labels):
    # 使用医学领域预训练BERT
    if DATA_TYPE == 'bimcv':
        MODEL_NAME = 'dccuchile/bert-base-spanish-wwm-cased'
    else:
        MODEL_NAME = 'emilyalsentzer/Bio_ClinicalBERT'
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # 分析类别分布
    label_counts = Counter(labels)
    # 计算类别权重以处理不平衡
    weights = [1.0, 1.0]  # 默认权重
    if label_counts[0] > 0 and label_counts[1] > 0:
        weights = [1.0 / label_counts[0], 1.0 / label_counts[1]]
        weights = torch.tensor(weights, dtype=torch.float).to(device)
        print(f"Using class weights: {weights.cpu().numpy()}")
    else:
        weights = None

    # 创建数据集
    dataset = MedicalReportDataset(reports, labels, tokenizer, max_len=256)
    # 使用分层抽样保持类别比例
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1/3, random_state=42)
    indices = list(range(len(dataset)))

    for train_idx, test_idx in sss.split(indices, labels):
        train_indices = train_idx
        test_indices = test_idx

    # 创建训练集和测试集
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # 创建数据加载器
    BATCH_SIZE = 16
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 初始化基础BERT模型
    base_model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    )

    # 使用自定义加权模型
    model = WeightedMedicalBERT(base_model, findings_weight=0.7, impression_weight=0.3).to(device)

    # 训练参数
    EPOCHS = 5
    LEARNING_RATE = 2e-5

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False)

    # 使用加权交叉熵损失
    if weights is not None:
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # 训练循环
    model = train_model(model, train_loader, test_loader, optimizer, criterion, EPOCHS)

    # 评估模型
    y_true, y_pred = evaluate_model(model, test_loader)

    # 计算指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # 分类报告和混淆矩阵
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Non-Pneumonia', 'Pneumonia']))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print(f'\nTest Metrics:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    return model


# 使用示例
if __name__ == "__main__":
    _, reports, labels = get_data()
    model = main(reports, labels)