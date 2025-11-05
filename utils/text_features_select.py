from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
# 新增的特征选择和降维库
from sklearn.feature_selection import SelectKBest, SelectFromModel, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import numpy as np

def select_features_mutual_info(features, labels):
    """使用互信息选择前k个最佳特征"""
    k = find_optimal_dimensionality(features, labels,'mutual_info')
    print(f"交叉验证确定{k}个最佳特征...")
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    selector.fit(features, labels)
    return selector

def select_features_rf(features, labels, threshold='mean'):
    """使用随机森林进行特征选择"""
    print("使用随机森林进行特征选择...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(features, labels)
    selector = SelectFromModel(rf, threshold=threshold)
    return selector

def select_features_rfe(features, labels):
    """使用递归特征消除(RFE)进行特征选择"""
    estimator = SVC(kernel="linear", random_state=42)
    n_features = find_optimal_dimensionality(features, labels,'rfe')
    print(f"交叉验证确定{n_features}个最佳特征...")
    selector = RFE(estimator, n_features_to_select=n_features, step=10)
    selector.fit(features, labels)
    return selector

def reduce_dim_pca(features, labels, n_components=None):
    """使用PCA进行特征降维"""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    n_components = n_components or find_optimal_dimensionality(features, labels,'pca')
    print(f"确定{n_components}个最佳特征...")
    pca = PCA(n_components=n_components)
    pca.fit(features_scaled)
    return pca

def reduce_dim_umap(features, labels):
    """使用UMAP进行非线性降维"""
    n_components = find_optimal_dimensionality(features, labels,'umap')
    print(f"交叉验证确定{n_components}个最佳特征...")
    reducer = UMAP(n_components=n_components, random_state=42)
    reducer.fit(features)
    return reducer

def reduce_dim_tsne(features, n_components=2):
    """使用t-SNE进行非线性降维"""
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=30)
    tsne.fit(features)
    return tsne


def find_optimal_dimensionality(features, labels, method='pca'):
    """通过交叉验证寻找最佳维度"""
    # 设置候选维度
    candidates = []
    if method == 'pca':
        candidates = [64, 128, 192, 256, 320, 384, 512]
    elif method == 'rfe':
        candidates = [64, 128, 192, 256, 320, 384, 512]
    elif method == 'umap':
        candidates = [5, 10, 15, 20, 30, 50]
    elif method == 'mutual_info':
        candidates = [64, 128, 192, 256, 320, 384, 512]

    best_score = 0
    best_param = None

    for param in candidates:
        if method == 'pca':
            features_opt = PCA(n_components=param).fit_transform(features)
        elif method == 'rfe':
            estimator = SVC(kernel="linear", random_state=42)
            features_opt = RFE(estimator, n_features_to_select=param, step=10).fit_transform(features, labels)
        elif method == 'umap':
            features_opt = UMAP(n_components=param, random_state=42).fit_transform(features)
        elif method == 'mutual_info':
            features_opt = SelectKBest(score_func=mutual_info_classif, k=param).fit_transform(features, labels)

        # 5折交叉验证
        scores = cross_val_score(
            SVC(), features_opt, labels, cv=5, scoring='accuracy'
        )
        mean_score = np.mean(scores)

        if mean_score > best_score:
            best_score = mean_score
            best_param = param
    print('最佳维度：',best_param)
    return best_param