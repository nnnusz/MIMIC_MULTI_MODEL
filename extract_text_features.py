import torch
import h5py
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import os
import numpy as np
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from utils.text_features_select import *
import time

# 加载不同数据源
DATA_TYPE = 'mimic'
from mimic_data_loader import get_data

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 自定义加权模型类 - 与训练代码保持一致
class WeightedMedicalBERT(torch.nn.Module):
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
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(weighted_logits.view(-1, self.bert.config.num_labels), labels.view(-1))

        return {
            'loss': loss,
            'logits': weighted_logits,
            'findings_hidden_states': findings_outputs.hidden_states,
            'impression_hidden_states': impression_outputs.hidden_states
        }


# 自定义数据集类（用于特征提取）- 修改为处理findings和impression分开
class MedicalReportFeatureDataset(Dataset):
    def __init__(self, reports, labels, tokenizer, max_len):
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


def extract_features(reports, labels, model_path, output_file, max_len, batch_size):
    """
    从训练好的加权模型中提取特征并保存到HDF5文件

    参数:
    reports: 报告文本列表
    labels: 对应标签列表
    model_path: 训练好的模型路径 (.pth文件)
    output_file: 输出的HDF5文件路径
    max_len: 最大序列长度
    batch_size: 批处理大小
    """
    # 确定模型类型
    if DATA_TYPE == 'bimcv':
        MODEL_NAME = 'dccuchile/bert-base-spanish-wwm-cased'
    else:
        MODEL_NAME = 'emilyalsentzer/Bio_ClinicalBERT'
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # 初始化基础BERT模型
    base_model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        output_attentions=False,
        output_hidden_states=True  # 需要隐藏状态用于特征提取
    )

    # 创建加权模型
    model = WeightedMedicalBERT(base_model, findings_weight=0.7, impression_weight=0.3)

    # 加载微调后的模型权重
    if model_path is not None and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)

        # 检查模型架构是否匹配
        if 'model_state_dict' in checkpoint:
            # 加载完整模型状态
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"Loaded model weights from {model_path}")
        else:
            # 直接加载模型权重
            model.load_state_dict(checkpoint)
            print(f"Loaded model weights from {model_path}")
    else:
        print(f"Model path {model_path} does not exist, using pre-trained weights only")

    model = model.to(device)
    model.eval()  # 设置为评估模式

    # 创建数据集和数据加载器
    dataset = MedicalReportFeatureDataset(reports, labels, tokenizer, max_len=max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 初始化特征存储
    all_features = []
    all_labels = []
    all_reports = []

    print(f"Extracting features for {len(dataset)} reports...")

    # 提取特征
    with torch.no_grad():
        for batch in tqdm(dataloader):
            findings_input_ids = batch['findings_input_ids'].to(device)
            findings_attention_mask = batch['findings_attention_mask'].to(device)
            impression_input_ids = batch['impression_input_ids'].to(device)
            impression_attention_mask = batch['impression_attention_mask'].to(device)
            batch_labels = batch['labels'].cpu().numpy()

            # 获取模型输出
            outputs = model(
                findings_input_ids=findings_input_ids,
                findings_attention_mask=findings_attention_mask,
                impression_input_ids=impression_input_ids,
                impression_attention_mask=impression_attention_mask
            )

            # 使用[CLS]标记作为整个序列的表示
            # 获取findings和impression的CLS标记
            findings_cls = outputs['findings_hidden_states'][-1][:, 0, :].cpu().numpy()
            impression_cls = outputs['impression_hidden_states'][-1][:, 0, :].cpu().numpy()

            # 加权融合特征
            weighted_features = 0.7 * findings_cls + 0.3 * impression_cls

            all_features.append(weighted_features)
            all_labels.append(batch_labels)

            # 获取当前批次的报告文本
            start_idx = len(all_reports)
            end_idx = start_idx + len(weighted_features)
            all_reports.extend(reports[start_idx:end_idx])

    # 合并所有特征和标签
    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)

    return features, labels


def save_features(features, labels, texts, name):
    """保存特征和原文到HDF5文件"""
    os.makedirs('output', exist_ok=True)
    filename = f"output/{DATA_TYPE}_{name}_text_features.h5"
    with h5py.File(filename, 'w') as hf:
        hf.create_dataset('features', data=features)
        hf.create_dataset('labels', data=labels)
    print(f"Features saved to {filename}")
    pathfilename = f"output/{DATA_TYPE}_report_text.txt"
    with open(pathfilename, 'w') as f:
        f.write('###'.join(texts))


def classify(X_train, X_test, y_train, y_test):
    # 创建SVC模型（关键参数调整）
    svc_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

    # 训练模型
    svc_model.fit(X_train, y_train)

    # 预测测试集
    y_pred = svc_model.predict(X_test)

    # 评估指标计算
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # 提取二分类指标
    tn, fp, fn, tp = conf_matrix.ravel()
    precision = tp / (tp + fp)  # 精确度（正类预测准确率）
    recall = tp / (tp + fn)  # 召回率（正类覆盖率）
    f1 = 2 * (precision * recall) / (precision + recall)  # F1分数

    # 打印完整评估报告
    print("=" * 60)
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确度 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数 (F1-score): {f1:.4f}")


def main():
    _, reports, labels = get_data()

    # 记录执行时间
    t_start = []
    t_end = []

    # 划分训练集测试集
    X_train, X_test, y_train, y_test = train_test_split(
        reports, labels, test_size=0.2, stratify=labels, random_state=42
    )
    texts = np.concatenate((X_train, X_test))
    labels = np.concatenate((y_train, y_test))
    # 提取特征
    model_path = f'../single/{DATA_TYPE}_text_model.pth'
    features_path = f'output/{DATA_TYPE}_text_features.h5'
    print('提取起始特征')
    t_start.append(time.time())
    train_features, train_labels = extract_features(
        reports=X_train,
        labels=y_train,
        model_path=model_path,
        output_file=features_path,
        max_len=256,
        batch_size=64
    )
    test_features, test_labels = extract_features(
        reports=X_test,
        labels=y_test,
        model_path=model_path,
        output_file=features_path,
        max_len=256,
        batch_size=64
    )
    t_end.append(time.time())
    print('起始特征形状：',train_features.shape)

    print('评估起始特征')
    classify(train_features, test_features, train_labels, test_labels)

    print('互信息MI')
    t_start.append(time.time())
    mi = select_features_mutual_info(train_features, train_labels)
    train_mi = mi.transform(train_features)
    test_mi = mi.transform(test_features)
    t_end.append(time.time())
    print('评估MI')
    mi_features = np.concatenate((train_mi, test_mi))
    classify(train_mi, test_mi, train_labels, test_labels)
    save_features(mi_features, labels, texts, 'mi')

    print('随机森林')
    t_start.append(time.time())
    rf = select_features_rf(train_features, train_labels)
    train_rf = rf.transform(train_features)
    test_rf = rf.transform(test_features)
    t_end.append(time.time())
    print('评估随机森林')
    rf_features = np.concatenate((train_rf, test_rf))
    classify(train_rf, test_rf, train_labels, test_labels)
    save_features(rf_features, labels, texts, 'rf')

    print('递归特征消除')
    t_start.append(time.time())
    rfe = select_features_rfe(train_features, train_labels)
    train_rfe = rfe.transform(train_features)
    test_rfe = rfe.transform(test_features)
    t_end.append(time.time())
    print('评估RFE')
    rfe_features = np.concatenate((train_rfe, test_rfe))
    classify(train_rfe, test_rfe, train_labels, test_labels)
    save_features(rfe_features, labels, texts, 'rfe')

    print('PCA降维')
    t_start.append(time.time())
    pca = reduce_dim_pca(train_features, train_labels)
    train_pca = pca.transform(train_features)
    test_pca = pca.transform(test_features)
    t_end.append(time.time())
    print('PCA降维形状：', test_pca.shape)
    print('评估PCA')
    pca_features = np.concatenate((train_pca, test_pca))
    classify(train_pca, test_pca, train_labels, test_labels)
    save_features(pca_features, labels, texts, 'pca')

    print('UMAP降维')
    t_start.append(time.time())
    umap = reduce_dim_umap(train_features, train_labels)
    train_umap = umap.transform(train_features)
    test_umap = umap.transform(test_features)
    t_end.append(time.time())
    print('UMAP降维形状：', test_umap.shape)
    print('评估UMAP')
    umap_features = np.concatenate((train_umap, test_umap))
    classify(train_umap, test_umap, train_labels, test_labels)
    save_features(umap_features, labels, texts, 'umap')

    print('算法执行时间：')
    for i in range(len(t_start)):
        print(t_end[i] - t_start[i], '\n')


if __name__ == "__main__":
    main()