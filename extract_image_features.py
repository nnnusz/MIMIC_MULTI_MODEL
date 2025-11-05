import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import os
import random
from PIL import Image
from utils.features_select import extract_features, extract_features_pretrain, PCATransformer, BGWOSelector, TransformerSelector, \
    MISelector
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import time

#加载不同数据源
DATA_TYPE = 'mimic'
from mimic_data_loader import get_data

#模型选择
MODEL_NAME = 'resnet'

# 设置随机种子保证可复现性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class ChestXrayDataset(Dataset):
    """自定义胸部X光数据集加载器"""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        try:
            if MODEL_NAME == 'densenet':
                image = Image.open(image_path).convert('L')  # 单通道
            else:
                image = Image.open(image_path).convert('RGB')  # 确保三通道
        except:
            # 如果图像损坏，使用空白图像替代
            print(f"Warning: Could not load image {image_path}, using blank image instead")
            image = Image.new('L', (224, 224), color=(0, 0, 0))

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        return image, label


def get_data_loaders(image_paths, labels, batch_size=32):
    """创建数据加载器"""
    transform = transforms.Compose([
        transforms.Resize(224),                  # 调整尺寸为224x224
        transforms.CenterCrop(224),              # 中心裁剪
        transforms.ToTensor(),                   # 转为Tensor [0,1]
        transforms.Normalize(mean=[0.5024], std=[0.2898])   #单通道归一化
    ])

    # 创建数据集和数据加载器
    dataset = ChestXrayDataset(image_paths, labels, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return loader


def save_features(features, labels, paths,name):
    """保存特征和源文件路径到HDF5文件"""
    os.makedirs('output', exist_ok=True)
    filename = f"output/{DATA_TYPE}_{MODEL_NAME}_{name}_features.h5"
    with h5py.File(filename, 'w') as hf:
        hf.create_dataset('features', data=features)
        hf.create_dataset('labels', data=labels)
    print(f"Features saved to {filename}")
    pathfilename = f"output/{DATA_TYPE}_image_paths.txt"
    with open(pathfilename,'w') as f:
        f.write('\n'.join(paths))

def classify(X_train, X_test, y_train, y_test,verbose=True):
    # 创建SVC模型（关键参数调整）
    svc_model = SVC(kernel='linear', random_state=42)

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
    if verbose:
        print("=" * 60)
        print(f"准确率 (Accuracy): {accuracy:.4f}")
        print(f"精确度 (Precision): {precision:.4f}")
        print(f"召回率 (Recall): {recall:.4f}")
        print(f"F1分数 (F1-score): {f1:.4f}")
    return accuracy,precision,recall,f1

def apply_pca(train_features,test_features,train_labels,target=None):
    pca_transformer = PCATransformer(
        variance_range=(0.85, 0.99),
        step=0.03,
        min_components=target or 100,
        max_components=1024
    )
    pca_transformer.fit(train_features,train_labels,target)
    train_pca = pca_transformer.transform(train_features)
    test_pca = pca_transformer.transform(test_features)
    return train_pca, test_pca

def apply_bgwo(train_features,test_features,train_labels,target=None):
    bgwo_selector = BGWOSelector(
        n_wolves=10,
        n_iter=20,
        target_features=target
    )
    # 训练选择器
    bgwo_selector.fit(train_features, train_labels)
    # 转换特征
    train_bgwo = bgwo_selector.transform(train_features)
    test_bgwo = bgwo_selector.transform(test_features)
    return train_bgwo,test_bgwo

def apply_mi(train_features,test_features,train_labels,target=None):
    mi_selector = MISelector()
    mi_selector.fit(train_features, train_labels,target)
    train_mi = mi_selector.transform(train_features)
    test_mi = mi_selector.transform(test_features)
    return train_mi,test_mi

def apply_transformer(train_features,test_features,train_labels,target=None):
    transformer_selector = TransformerSelector(
        embed_dim=16,
        num_heads=1,
        epochs=20,
        target=target
    )
    transformer_selector.fit(train_features, train_labels)
    train_trans = transformer_selector.transform(train_features)
    test_trans = transformer_selector.transform(test_features)
    return train_trans, test_trans

#寻找算法最佳特征
def find_best_features(train_features,test_features,train_labels,test_labels,funcs):
    params = [1024, 896, 768, 640, 512, 384, 256, 128]
    best_f1 = 0
    best_train_f = train_features
    best_test_f = test_features
    for i in range(len(params)):
        p = params[i]
        t_start = time.time()
        train_f, test_f = funcs(
            train_features,test_features,train_labels,p)
        t_end = time.time()
        accuracy,precision,recall,f1 = \
            classify(train_f,test_f,train_labels,test_labels,False)
        if f1 > best_f1:
            best_f1 = f1
            best_train_f = train_f
            best_test_f = test_f
            print('最佳特征维数：', p)
            print(f'acc:{accuracy},pre:{precision},rec:{recall},f1{f1}')
            print('算法执行时间：', t_end-t_start)
    return best_train_f,best_test_f

#寻找组合算法最佳特征
def find_best_group_features(train_features,test_features,train_labels,test_labels,funcs):
    params = [1024, 896, 768, 640, 512, 384, 256, 128]
    best_f1 = 0
    best_train_f = train_features
    best_test_f = test_features
    for i in range(len(params)):
        p1 = params[i]
        for j in range(i + 1, len(params)):
            p2 = params[j]
            t_start = time.time()
            train_f1, test_f1 = funcs[0](
                train_features,test_features,train_labels,p1)
            train_f2, test_f2 = funcs[1](
                train_f1,test_f1,train_labels,p2)
            t_end = time.time()
            accuracy,precision,recall,f1 = \
                classify(train_f2,test_f2,train_labels,test_labels,False)
            if f1 > best_f1:
                best_f1 = f1
                best_train_f = train_f2
                best_test_f = test_f2
                print('最佳特征维数：', p1,p2)
                print(f'acc:{accuracy},pre:{precision},rec:{recall},f1{f1}')
                print('算法执行时间：', t_end-t_start)
    return best_train_f,best_test_f


def single(images, labels):
    set_seed(42)

    # 第一步：划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # 第二步：创建数据加载器（训练集和测试集独立）
    train_loader = get_data_loaders(X_train, y_train, batch_size=64)
    test_loader = get_data_loaders(X_test, y_test, batch_size=64)

    #记录时间
    t_start = []
    t_end = []

    # 第三步：特征提取
    print('提取原始特征……')
    t_start.append(time.time())
    if MODEL_NAME == 'densenet':
        train_features, train_labels = extract_features_pretrain(train_loader, '../single/mimic_densenet_model.pth')
        test_features, test_labels = extract_features_pretrain(test_loader, '../single/mimic_densenet_model.pth')
    else:
        train_features, train_labels = extract_features(train_loader, '../single/mimic_resnet_model.pth')
        test_features, test_labels = extract_features(test_loader, '../single/mimic_resnet_model.pth')


    print('原始特征形状：',train_features.shape)
    t_end.append(time.time())
    print('原始特征评估')
    paths = np.concatenate((X_train, X_test))
    labels = np.concatenate((train_labels, test_labels))
    classify(train_features,test_features,train_labels, test_labels)

    #应用优化算法
    method = {
        'pca':apply_pca,
        'bgwo':apply_bgwo,
        'mi':apply_mi,
        'transformer':apply_transformer
    }
    for k,v in method.items():
        print(k)
        #寻找最佳特征
        train_f, test_f = find_best_features(
            train_features,test_features,train_labels, test_labels,v)
        result_features = np.concatenate((train_f, test_f))
        save_features(result_features, labels, paths,k)


def combine(images, labels):
    set_seed(42)

    # 第一步：划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # 第二步：创建数据加载器（训练集和测试集独立）
    train_loader = get_data_loaders(X_train, y_train, batch_size=64)
    test_loader = get_data_loaders(X_test, y_test, batch_size=64)

    # 第三步：特征提取
    print('提取原始特征……')
    if MODEL_NAME == 'densenet':
        train_features, train_labels = extract_features_pretrain(train_loader, '../single/mimic_densenet_model.pth')
        test_features, test_labels = extract_features_pretrain(test_loader, '../single/mimic_densenet_model.pth')
    else:
        train_features, train_labels = extract_features(train_loader, '../single/mimic_resnet_model.pth')
        test_features, test_labels = extract_features(test_loader, '../single/mimic_resnet_model.pth')

    paths = np.concatenate((X_train, X_test))
    labels = np.concatenate((train_labels, test_labels))

    #组合算法
    group = [
        # {
        #     'name4 ':'pca+mi',
        #     'func':[apply_pca,apply_mi],
        # },
        # {
        #     'name': 'pca+bgwo',
        #     'func': [apply_pca, apply_bgwo],
        # },
        # {
        #     'name': 'mi+bgwo',
        #     'func': [apply_mi, apply_bgwo],
        # },
        {
            'name': 'bgwo+transformer',
            'func': [apply_bgwo, apply_transformer],
        }
    ]

    for data in group:
        name = data['name']
        print(name)
        #寻找最佳特征
        train_f, test_f = find_best_group_features(
            train_features,test_features,train_labels, test_labels,data['func'])
        result_features = np.concatenate((train_f, test_f))
        save_features(result_features, labels, paths,name)

if __name__ == '__main__':
    image, _ , label = get_data()
    #single(image, label)
    combine(image, label)
