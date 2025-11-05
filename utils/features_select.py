import random
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import confusion_matrix
import matplotlib
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, TransformerMixin

matplotlib.use('Agg')  # 使用Agg后端生成图像文件而不显示
import torch.nn as nn
import torchvision.models as models
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torchxrayvision as xrv

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True

def extract_features(data_loader, model_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )

    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))

    model = nn.Sequential(*list(model.children())[:-1])  # 移除最后一层
    model = model.to(device).eval()

    features = []
    labels_list = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Extracting features"):
            images = images.to(device)
            outputs = model(images)
            # 调整特征维度
            outputs = outputs.view(outputs.size(0), -1)  # 使用view而不是squeeze更安全
            features.append(outputs.cpu())
            labels_list.append(labels)

    features = torch.cat(features).numpy()
    labels = torch.cat(labels_list).numpy()
    return features, labels

class CustomDenseNet(xrv.models.DenseNet):
    def __init__(self, weights="densenet121-res224-chex", apply_sigmoid=False, op_threshs=None):
        super().__init__(weights=weights, apply_sigmoid=apply_sigmoid, op_threshs=op_threshs)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return self.classifier(out)

def extract_features_pretrain(data_loader,model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomDenseNet(
        weights=None,  # 不加载预训练权重
        apply_sigmoid=False,
        op_threshs=None
    )
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    state_dict = torch.load(model_path)

    # 移除模型中不存在的键
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}

    # 加载过滤后的状态字典
    model.load_state_dict(filtered_state_dict)

    model = model.to(device).eval()
    # 获取特征提取器
    if hasattr(model, "features"):
        feature_extractor = model.features
    elif hasattr(model, "model") and hasattr(model.model, "features"):
        feature_extractor = model.model.features
    else:
        # 对于没有明确features属性的模型，使用整个模型
        feature_extractor = model
    # 创建特征提取器（移除分类层）
    feature_extractor = torch.nn.Sequential(
        feature_extractor,  # [B, 1024, 7, 7]
        torch.nn.ReLU(inplace=True),
        torch.nn.AdaptiveAvgPool2d(1)  # 全局平均池化 [B, 1024, 1, 1]
    )
    features = []
    labels_list = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Extracting features"):
            images = images.to(device)
            outputs = feature_extractor(images)
            # 调整特征维度
            feature = outputs.view(outputs.size(0), -1)  # 展平为 [B, 1024]
            features.append(feature.cpu())
            labels_list.append(labels)

    features = torch.cat(features).numpy()
    labels = torch.cat(labels_list).numpy()
    return features,labels

#PCA降维
class PCATransformer:
    def __init__(self, variance_range=(0.85, 0.99), step=0.03, min_components=128, max_components=1024):
        self.variance_range = variance_range
        self.step = step
        self.min_components = min_components
        self.max_components = max_components
        self.pca = None
        self.optimal_variance_ = None  # 存储最优方差值

    def find_optimal_variance(self, features, labels):
        """通过交叉验证寻找最优方差保留比例"""
        from sklearn.model_selection import StratifiedKFold
        from sklearn.svm import SVC

        best_f1 = -1
        best_variance = self.variance_range[0]
        X, y = features, labels

        for variance in np.arange(self.variance_range[0],
                                  self.variance_range[1],
                                  self.step):
            kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            f1_scores = []

            for train_idx, val_idx in kfold.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # 拟合PCA
                pca = PCA(n_components=variance, svd_solver='full')
                X_train_pca = pca.fit_transform(X_train)
                X_val_pca = pca.transform(X_val)

                # 训练简单分类器评估
                clf = SVC(kernel='linear', probability=True, random_state=42)
                clf.fit(X_train_pca, y_train)
                y_pred = clf.predict(X_val_pca)

                # 计算F1分数
                tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                f1_scores.append(f1)

            mean_f1 = np.mean(f1_scores)
            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_variance = variance

        self.optimal_variance_ = best_variance
        print('最优方差比例：',best_variance)
        return best_variance

    def fit(self, features, labels, n_components=0):
        """在训练数据上拟合PCA模型"""
        if not n_components is None and n_components >= 0:
            self.n_components = n_components
        else:
            self.n_components = self.find_optimal_variance(features,labels)
        #特征数量边界
        self.n_components = max(self.n_components,self.min_components)
        self.n_components = min(self.n_components, self.max_components)

        self.pca = PCA(n_components=self.n_components, svd_solver='auto')
        self.pca.fit(features)
        self.fitted_ = True
        return self

    def transform(self, features):
        """应用PCA转换"""
        if not self.fitted_:
            raise RuntimeError("PCA must be fitted before transformation")
        return self.pca.transform(features)

    def fit_transform(self, features,labels):
        """同时拟合和转换"""
        self.fit(features,labels)
        return self.transform(features)


class BGWOSelector:
    def __init__(self, n_wolves=10, n_iter=20, target_features=None,
                 verbose=False, device=None):
        """
        改进的灰狼优化特征选择器
        使用基于统计特性和特征相关性的适应度函数

        参数:
        n_wolves (int): 狼群数量
        n_iter (int): 迭代次数
        target_features (int): 目标特征数量
        verbose (bool): 是否显示详细输出
        device (str): 计算设备 ('cpu' 或 'cuda')
        """
        self.n_wolves = n_wolves
        self.n_iter = n_iter
        self.verbose = verbose
        self.target_features = target_features
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.selected_mask_ = None
        self.selected_indices_ = None
        self.feature_importances_ = None  # 存储特征重要性分数
        self.correlation_matrix = None
        self.class_separation = None

    def _compute_feature_statistics(self, features, labels):
        """预计算特征统计信息"""
        # 转换为PyTorch张量
        features_tensor = torch.tensor(features).float().to(self.device)
        labels_tensor = torch.tensor(labels).float().to(self.device)

        # 计算类间分离度
        class_0_mask = labels_tensor == 0
        class_1_mask = labels_tensor == 1

        mean_0 = torch.mean(features_tensor[class_0_mask], dim=0)
        mean_1 = torch.mean(features_tensor[class_1_mask], dim=0)

        std_0 = torch.std(features_tensor[class_0_mask], dim=0)
        std_1 = torch.std(features_tensor[class_1_mask], dim=0)

        # 避免除以零
        std_combined = torch.sqrt(std_0 ** 2 + std_1 ** 2) + 1e-8
        self.class_separation = torch.abs(mean_0 - mean_1) / std_combined

        # 计算特征相关性矩阵
        centered_features = features_tensor - torch.mean(features_tensor, dim=0)
        cov_matrix = torch.matmul(centered_features.t(), centered_features) / (features_tensor.shape[0] - 1)
        std_matrix = torch.outer(torch.std(features_tensor, dim=0), torch.std(features_tensor, dim=0)) + 1e-8
        self.correlation_matrix = cov_matrix / std_matrix

        # 计算特征方差
        self.feature_variances = torch.var(features_tensor, dim=0)

        return features_tensor, labels_tensor

    def _fitness_function(self, position, features, labels):
        """
        新型适应度函数：基于统计特性和特征相关性
        包含四个关键指标:
        1. 类间分离度 (class_separation)
        2. 特征相关性 (redundancy)
        3. 特征多样性 (diversity)
        4. 特征稳定性 (stability)
        """
        selected = position > 0.5
        selected_indices = torch.where(selected)[0]
        n_selected = len(selected_indices)

        # 如果没有选择特征或选择太少，返回低分
        if n_selected < 10:
            return -10.0

        # 1. 类间分离度 (越高越好)
        separation_score = torch.mean(self.class_separation[selected_indices]).item()

        # 2. 特征冗余度 (越低越好)
        # 计算选定特征之间的平均绝对相关性
        selected_corr = self.correlation_matrix[selected_indices][:, selected_indices]
        np.fill_diagonal(selected_corr.cpu().numpy(), 0)  # 忽略自相关
        redundancy = torch.mean(torch.abs(selected_corr)).item()

        # 3. 特征多样性 (越高越好)
        # 使用选定特征的方差分布来衡量多样性
        selected_variances = self.feature_variances[selected_indices]
        diversity = torch.var(selected_variances).item() / (torch.mean(selected_variances).item() + 1e-8)

        # 4. 特征稳定性 (越高越好)
        # 计算特征之间的互补性
        abs_corr = torch.abs(selected_corr)
        complementarity = 1.0 - (torch.sum(abs_corr) - n_selected) / (n_selected * (n_selected - 1) + 1e-8)

        # 组合分数 (加权平均)
        fitness = (
                0.4 * separation_score +  # 类间分离度权重最高
                0.3 * (1 - redundancy) +  # 低冗余很重要
                0.2 * diversity +  # 特征多样性
                0.1 * complementarity  # 特征互补性
        )

        # 惩罚选择过多特征
        if not self.target_features is None and n_selected > self.target_features * 1.5:
            excess_penalty = (n_selected - self.target_features) / self.target_features
            fitness *= max(0.5, 1.0 - excess_penalty * 0.5)

        return fitness

    def fit(self, features, labels):
        """训练特征选择器"""
        # 预计算特征统计信息
        features_tensor, labels_tensor = self._compute_feature_statistics(features, labels)
        n_features = features_tensor.shape[1]

        # 预过滤低方差特征
        variance_threshold = torch.quantile(self.feature_variances, 0.3).item()
        high_var_mask = self.feature_variances > variance_threshold
        high_var_indices = torch.where(high_var_mask)[0]

        if self.verbose:
            print(f"预过滤特征: {len(high_var_indices)}/{n_features} (方差阈值={variance_threshold:.4f})")

        # 初始化狼群 (只在高方差特征上初始化)
        wolves = torch.rand(self.n_wolves, len(high_var_indices), device=self.device)

        # 计算初始适应度
        fitness = torch.zeros(self.n_wolves, device=self.device)
        for i in range(self.n_wolves):
            # 创建完整特征位置向量
            full_position = torch.zeros(n_features, device=self.device)
            full_position[high_var_indices] = wolves[i]
            fitness[i] = self._fitness_function(full_position, features_tensor, labels_tensor)

        # 确定alpha, beta, delta狼
        sorted_idx = torch.argsort(fitness, descending=True)
        alpha_wolf = wolves[sorted_idx[0]].clone()
        beta_wolf = wolves[sorted_idx[1]].clone()
        delta_wolf = wolves[sorted_idx[2]].clone()
        alpha_score = fitness[sorted_idx[0]].item()

        # BGWO主循环
        for iter in range(self.n_iter):
            a = 2 - iter * (2 / self.n_iter)  # 线性递减

            for i in range(self.n_wolves):
                # 更新位置
                A1 = a * (2 * torch.rand(1, device=self.device) - 1)
                A2 = a * (2 * torch.rand(1, device=self.device) - 1)
                A3 = a * (2 * torch.rand(1, device=self.device) - 1)

                C1 = 2 * torch.rand(1, device=self.device)
                C2 = 2 * torch.rand(1, device=self.device)
                C3 = 2 * torch.rand(1, device=self.device)

                X1 = alpha_wolf - A1 * torch.abs(C1 * alpha_wolf - wolves[i])
                X2 = beta_wolf - A2 * torch.abs(C2 * beta_wolf - wolves[i])
                X3 = delta_wolf - A3 * torch.abs(C3 * delta_wolf - wolves[i])

                new_position = (X1 + X2 + X3) / 3

                # 二进制转换 (Sigmoid函数 + 随机扰动)
                wolves[i] = torch.sigmoid(new_position) + 0.1 * (torch.rand_like(new_position) - 0.5)
                wolves[i] = torch.clamp(wolves[i], 0, 1)

                # 计算新位置的适应度
                full_position = torch.zeros(n_features, device=self.device)
                full_position[high_var_indices] = wolves[i]
                fitness[i] = self._fitness_function(full_position, features_tensor, labels_tensor)

            # 更新头狼
            sorted_idx = torch.argsort(fitness, descending=True)
            current_best = fitness[sorted_idx[0]].item()

            if current_best > alpha_score:
                alpha_wolf = wolves[sorted_idx[0]].clone()
                beta_wolf = wolves[sorted_idx[1]].clone()
                delta_wolf = wolves[sorted_idx[2]].clone()
                alpha_score = current_best

            if self.verbose:
                avg_fitness = torch.mean(fitness).item()
                print(f"Iter {iter + 1}/{self.n_iter} | Best: {alpha_score:.4f} | Avg: {avg_fitness:.4f}")

        # 创建最终特征选择掩码
        full_alpha = torch.zeros(n_features, device=self.device)
        full_alpha[high_var_indices] = alpha_wolf

        # 选择前target_features个特征
        _, top_indices = torch.topk(full_alpha, k=self.target_features)
        self.selected_mask_ = torch.zeros(n_features, dtype=torch.bool, device=self.device)
        self.selected_mask_[top_indices] = True

        # 存储特征重要性
        self.feature_importances_ = full_alpha.cpu().numpy()

        if self.verbose:
            print(f"Selected {self.target_features} features")

        return self

    def transform(self, features):
        """应用特征选择"""
        if self.selected_mask_ is None:
            raise RuntimeError("Must fit the selector first")

        # 转换为numpy数组
        mask = self.selected_mask_.cpu().numpy()
        return features[:, mask]

    def get_support(self, indices=False):
        """获取选择的特征"""
        if indices:
            return np.where(self.selected_mask_.cpu().numpy())[0]
        return self.selected_mask_.cpu().numpy()


class MISelector:
    def __init__(self):
        """
        基于互信息的特征选择
        """
        self.selected_mask_ = None

    def fit(self, features, labels,target=None):
        """计算特征重要性"""
        # 计算互信息
        self.mi_scores = mutual_info_classif(features, labels, random_state=42)

        if target:
            k = target
        else:
            threshold = self.find_optimal_threshold(features, labels)
            # 选择前百分比的特征
            k = int(features.shape[1] * threshold)
        sorted_indices = np.argsort(self.mi_scores)[::-1]
        selected_indices = sorted_indices[:k]

        # 创建选择掩码
        self.selected_mask_ = np.zeros(features.shape[1], dtype=bool)
        self.selected_mask_[selected_indices] = True

        return self

    def transform(self, features):
        """应用特征选择"""
        if self.selected_mask_ is None:
            raise RuntimeError("Must fit the selector first")
        return features[:, self.selected_mask_]

    def find_optimal_threshold(self, features, labels):
        """通过交叉验证寻找最佳维度"""
        # 设置比例范围
        ratios = np.linspace(0.1, 0.7, 7)  # 测试10%-70%的特征比例

        best_score = 0
        best_param = 0.3  # 默认值

        for threshold in ratios:
            # 创建选择掩码
            k = int(features.shape[1] * threshold)
            sorted_indices = np.argsort(self.mi_scores)[::-1]
            selected_mask = np.zeros(features.shape[1], dtype=bool)
            selected_mask[sorted_indices[:k]] = True

            if np.sum(selected_mask) < 10:  # 确保至少选择10个特征
                continue

            features_subset = features[:, selected_mask]

            # 5折交叉验证
            scores = cross_val_score(
                SVC(kernel='linear', random_state=42),
                features_subset, labels, cv=3, scoring='f1_macro'
            )
            mean_score = np.mean(scores)

            if mean_score > best_score:
                best_score = mean_score
                best_param = threshold

        print(f'最佳特征比例：{best_param:.2f} (F1={best_score:.4f})')
        return best_param


class TransformerSelector(BaseEstimator, TransformerMixin):
    def __init__(self, embed_dim=16, num_heads=1, num_layers=1,
                 hidden_dim=32, dropout=0.1, learning_rate=1e-3,
                 epochs=30, batch_size=32,
                 device=None, verbose=False, patience=5,
                 focus_ratio=0.7, target=256,
                 stability_threshold=0.01):
        """
        简化高效的Transformer特征选择器

        关键优化：
        1. 模型结构简化：单层Transformer，减少头数和隐藏维度
        2. 专注特征选择：移除了分类任务头
        3. 预过滤机制：基于互信息预选特征范围
        4. 稳定性早停：监控特征重要性的变化
        """
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        self.patience = patience
        self.focus_ratio = focus_ratio
        self.target = target
        self.stability_threshold = stability_threshold

        self.model = None
        self.feature_importances_ = None
        self.selected_mask_ = None
        self.scaler = StandardScaler()
        self.loss_history = []
        self.importance_history = []
        self.top_idx = None  # 存储预选特征索引

    def _build_model(self, num_features):
        """构建简化特征选择模型"""

        class FeatureSelector(nn.Module):
            def __init__(self, num_features, embed_dim, num_heads, hidden_dim, dropout):
                super().__init__()
                self.num_features = num_features

                # 特征嵌入层
                self.embedding = nn.Linear(1, embed_dim)

                # 位置编码（可学习参数）
                self.positional_encoding = nn.Parameter(torch.randn(1, num_features, embed_dim) * 0.02)

                # 单层Transformer编码器
                self.encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim,
                    dropout=dropout,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

                # 特征评分器
                self.scorer = nn.Sequential(
                    nn.Linear(embed_dim, 1),
                    nn.Sigmoid()
                )

                # 初始化
                self._init_weights()

            def _init_weights(self):
                """权重初始化"""
                for module in self.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0.01)
                nn.init.normal_(self.positional_encoding, mean=0.0, std=0.01)

            def forward(self, x):
                # 输入形状: [batch_size, num_features]
                x = x.unsqueeze(-1)  # [batch_size, num_features, 1]

                # 特征嵌入
                embedded = self.embedding(x)  # [batch_size, num_features, embed_dim]
                embedded += self.positional_encoding

                # Transformer处理
                transformed = self.transformer(embedded)  # [batch_size, num_features, embed_dim]

                # 特征重要性评分
                scores = self.scorer(transformed).squeeze(-1)  # [batch_size, num_features]

                return scores

        return FeatureSelector(num_features, self.embed_dim, self.num_heads,
                               self.hidden_dim, self.dropout).to(self.device)

    def _prefilter_features(self, X, y):
        """基于互信息预选重要特征范围"""
        from sklearn.feature_selection import mutual_info_classif

        if self.verbose:
            print("执行特征预过滤...")

        # 计算互信息分数
        mi_scores = mutual_info_classif(X, y)

        # 选择顶部特征
        n_selected = max(10, int(X.shape[1] * self.focus_ratio))
        self.top_idx = np.argsort(mi_scores)[-n_selected:]

        if self.verbose:
            print(f"预选 {len(self.top_idx)}/{X.shape[1]} 个特征")

        return X[:, self.top_idx]

    def fit(self, X, y):
        """训练特征选择器"""
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)

        # 特征预过滤
        X_filtered = self._prefilter_features(X_scaled, y)

        # 转换为Tensor
        X_tensor = torch.FloatTensor(X_filtered).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        # 初始化模型
        num_features = X_filtered.shape[1]
        self.model = self._build_model(num_features)

        # 优化器
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)

        # 损失函数
        def feature_loss(scores):
            """特征稀疏性损失 + 多样性损失"""
            # 稀疏性：鼓励选择少量特征
            sparsity_loss = torch.mean(scores)

            # 多样性：鼓励特征间重要性差异
            diversity_loss = -torch.var(scores, dim=1).mean()

            return sparsity_loss + 0.5 * diversity_loss

        # 训练循环
        best_loss = float('inf')
        best_imp = None
        stability_counter = 0
        last_importance = None

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0

            # 随机批次训练
            indices = torch.randperm(X_tensor.size(0))
            for i in range(0, len(indices), self.batch_size):
                batch_idx = indices[i:i + self.batch_size]
                batch_x = X_tensor[batch_idx]

                optimizer.zero_grad()

                # 前向传播
                scores = self.model(batch_x)

                # 计算损失
                loss = feature_loss(scores)

                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()

            # 记录损失
            avg_loss = epoch_loss / (len(indices) / self.batch_size)
            self.loss_history.append(avg_loss)

            # 计算特征重要性
            self.model.eval()
            with torch.no_grad():
                all_scores = []
                for i in range(0, len(X_tensor), self.batch_size):
                    batch = X_tensor[i:i + self.batch_size]
                    batch_scores = self.model(batch)
                    all_scores.append(batch_scores.cpu())

                importance = torch.cat(all_scores).mean(dim=0).numpy()
                self.importance_history.append(importance)

            # 早停机制：检查特征稳定性
            if last_importance is not None:
                imp_change = np.mean(np.abs(importance - last_importance))
                if imp_change < self.stability_threshold:
                    stability_counter += 1
                    if stability_counter >= self.patience:
                        if self.verbose:
                            print(f"早停于轮次 {epoch + 1}: 特征稳定性达到 (变化={imp_change:.4f})")
                        break
                else:
                    stability_counter = 0

            last_importance = importance

            # 更新最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_imp = importance
                best_state = self.model.state_dict().copy()

            # 打印进度
            if self.verbose and (epoch + 1) % 5 == 0:
                print(f"轮次 {epoch + 1}/{self.epochs} | 损失: {avg_loss:.4f} | "
                      f"平均重要性: {np.mean(importance):.4f}")

        # 加载最佳模型
        self.model.load_state_dict(best_state)
        self.feature_importances_ = best_imp

        # 创建特征选择掩码（选择前K个特征）
        k = self.target
        top_indices = np.argsort(self.feature_importances_)[-k:]

        # 映射回原始特征空间
        self.selected_mask_ = np.zeros(X.shape[1], dtype=bool)
        self.selected_mask_[self.top_idx[top_indices]] = True

        if self.verbose:
            n_selected = np.sum(self.selected_mask_)
            print(f"最终选择 {n_selected}/{X.shape[1]} 个特征")
            print(f"重要性范围: {np.min(self.feature_importances_):.4f}-{np.max(self.feature_importances_):.4f}")

        return self

    def transform(self, X):
        """应用特征选择"""
        if self.selected_mask_ is None:
            raise RuntimeError("必须先训练选择器")

        # 标准化输入
        X_scaled = self.scaler.transform(X)
        return X_scaled[:, self.selected_mask_]

    def get_support(self, indices=False):
        """获取选择的特征"""
        if indices:
            return np.where(self.selected_mask_)[0]
        return self.selected_mask_
