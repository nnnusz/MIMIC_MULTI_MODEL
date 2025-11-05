import sys
import os
import time

from torch import optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import warnings
from utils.fusion_func import *

DATA_TYPE = 'mimic'

# 忽略警告
warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 特征文件位置
features_path = '../extract_features/output/'
image_feature_name = 'bgwo+transformer'
text_feature_name = 'rfe'
IMAGE_F = features_path + f'{DATA_TYPE}_resnet_{image_feature_name}_features.h5'
TEXT_F = features_path + f'{DATA_TYPE}_{text_feature_name}_text_features.h5'


# 2. 交叉注意力融合模型
class CrossAttnFusionModel(nn.Module):
    """交叉注意力融合模型"""
    def __init__(self, image_dim, text_dim, num_classes, hidden_dim=256):
        super().__init__()
        # 特征重加权
        self.reweight = FeatureReweighting(text_dim, image_dim)
        # 文本特征增强
        self.text_enhancer = TextFeatureEnhancer(text_dim, hidden_dim)
        # 图像特征投影
        self.image_projection = nn.Linear(image_dim, hidden_dim)
        # 交叉注意力融合
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,
            dropout = 0.5
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        # 多模态分类头
        self.multimodal_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        # 文本分类头
        self.text_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        # 存储注意力权重
        self.attention_weights = None

    def forward(self, image, text):
        # 特征重加权
        text, image = self.reweight(text, image)
        # 文本特征增强
        text_feat = self.text_enhancer(text)
        # 图像特征投影
        image_feat = self.image_projection(image)
        # 交叉注意力融合
        text_query = text_feat.unsqueeze(1)
        image_kv = image_feat.unsqueeze(1)
        attn_output, attn_weights = self.cross_attn(text_query, image_kv, image_kv)
        attn_output = attn_output.squeeze(1)

        # 存储注意力权重
        self.attention_weights = attn_weights
        # 残差连接和归一化
        fused_feat = self.attn_norm(text_feat + attn_output)
        # 文本分类结果
        text_logits = self.text_classifier(text_feat)
        # 多模态融合分类
        combined_feat = torch.cat([text_feat, fused_feat], dim=1)
        multimodal_logits = self.multimodal_classifier(combined_feat)
        return multimodal_logits, text_logits

    def get_attention_weights(self):
        """获取注意力权重"""
        return self.attention_weights

class FeatureReweighting(nn.Module):
    """特征重加权模块"""
    def __init__(self, text_dim, image_dim):
        super().__init__()
        self.text_gate = nn.Sequential(
            nn.Linear(text_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.image_gate = nn.Sequential(
            nn.Linear(image_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, text_feat, image_feat):
        text_weight = self.text_gate(text_feat)
        image_weight = self.image_gate(image_feat)
        text_feat = text_feat * (1 + text_weight)
        image_feat = image_feat * image_weight
        return text_feat, image_feat

class TextFeatureEnhancer(nn.Module):
    """文本特征增强模块"""
    def __init__(self, text_dim, hidden_dim=256):
        super().__init__()
        self.enhancer = nn.Sequential(
            nn.Linear(text_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )

    def forward(self, text_feat):
        return self.enhancer(text_feat)


# 4. 主函数
def main(num_epochs = 20, enhancement_factor=0.5):
    # 加载数据
    try:
        image_features, text_features, labels = load_h5_data(
            image_path=IMAGE_F,
            text_path=TEXT_F
        )
    except Exception as e:
        print(f"加载数据失败: {e}")
        sys.exit(1)

    print(f"图像特征形状: {image_features.shape}")
    print(f"文本特征形状: {text_features.shape}")
    print(f"标签形状: {labels.shape}")

    # 文本特征增强
    text_features = enhance_text_features(text_features, labels, enhancement_factor=enhancement_factor)

    # 检查标签分布
    unique_labels, counts = np.unique(labels, return_counts=True)
    num_classes = len(unique_labels)
    class_names = [f"Class {i}" for i in unique_labels]

    print("\n类别分布:")
    for label, count in zip(unique_labels, counts):
        print(f"类别 {label}: {count} 样本 ({count / len(labels) * 100:.2f}%)")

    # 划分数据集
    X_img_train, X_img_test, X_txt_train, X_txt_test, y_train, y_test = train_test_split(
        image_features, text_features, labels, test_size=0.2, stratify=labels, random_state=42
    )

    print(f"\n训练集大小: {len(y_train)}")
    print(f"测试集大小: {len(y_test)}")

    # 创建数据集和数据加载器
    train_dataset = MultimodalDataset(X_img_train, X_txt_train, y_train)
    test_dataset = MultimodalDataset(X_img_test, X_txt_test, y_test)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 初始化模型（交叉注意力融合）
    model = CrossAttnFusionModel(
        image_dim=image_features.shape[1],
        text_dim=text_features.shape[1],
        num_classes=num_classes,
        hidden_dim=256
    ).to(device)

    # 损失函数和优化器
    criterion = GuidedLoss(alpha=0.75)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # 训练参数
    best_val_f1 = 0.0
    best_val_auc = 0.0
    best_val_aupr = 0.0
    best_val_spec = 0.0
    train_losses, val_losses = [], []

    # 结果目录
    os.makedirs('results/cross', exist_ok=True)

    # 训练和评估（与concat_fusion.py相同）
    print("\n开始训练...")
    start_t = time.time()
    for epoch in range(num_epochs):
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, 'train'
        )
        train_losses.append(train_loss)

        # 验证
        val_results = evaluate_detail(model, test_loader, criterion, device)
        val_loss = val_results['loss']
        val_losses.append(val_loss)
        val_f1 = val_results['f1']

        # 更新学习率
        scheduler.step(val_loss)

        print(f"\nEpoch {epoch + 1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_auc = val_results['auc_score']
            best_val_aupr = val_results['aupr_score']
            best_val_spec = val_results['specificity']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_loss': val_loss,
            }, f'results/cross/{DATA_TYPE}_cross_model.pth')
            print(f"  保存最佳模型，F1分数: {val_f1:.4f}")
    end_t = time.time()
    print('训练时长:',end_t-start_t)
    # 绘制训练曲线
    plot_training_curves(
        train_losses, val_losses,
        f'results/cross/{DATA_TYPE}_training_curve.png'
    )

    # 加载最佳模型并在测试集上评估
    checkpoint = torch.load(f'results/cross/{DATA_TYPE}_cross_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    test_results = evaluate_detail(model, test_loader, criterion, device)

    # 测试集混淆矩阵
    plot_confusion_matrix(
        test_results['labels'],
        test_results['preds'],
        class_names,
        f'results/cross/{DATA_TYPE}_confusion_matrix.png'
    )

    # 保存结果到results/cross目录
    with open(f'results/cross/{DATA_TYPE}_report.txt', 'w',encoding='utf-8') as f:
        f.write("交叉注意力融合分类结果\n")
        f.write("=" * 50 + "\n")
        f.write(f"测试集大小: {len(y_test)}\n")
        f.write(f"准确率: {test_results['accuracy']:.4f}\n")
        f.write(f"精确率: {test_results['precision']:.4f}\n")
        f.write(f"召回率: {test_results['recall']:.4f}\n")
        f.write(f"F1分数: {test_results['f1']:.4f}\n")
        f.write(f"最佳验证F1分数: {best_val_f1:.4f}\n")
        f.write(f"AUC: {best_val_auc:.4f}\n")
        f.write(f"AUPR: {best_val_aupr:.4f}\n")
        f.write(f"specificity: {best_val_spec:.4f}\n")

    print("\n测试结果:")
    print(f"准确率: {test_results['accuracy']:.4f}")
    print(f"F1分数: {test_results['f1']:.4f}")
    print(f"最佳验证F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    # for i in range(10):
    #     enhancement_factor = 0.1 + i * 0.1
    #     print('文本增强因子：',enhancement_factor)
    #     main(enhancement_factor=enhancement_factor)
    main(enhancement_factor=0.0)