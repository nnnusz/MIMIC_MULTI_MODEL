import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import os


def extract_keywords_from_single_report(report, n_keywords=10):
    """
    从单个报告中提取关键词
    参数:
        report: 单个报告文本
        n_keywords: 提取的关键词数量
    返回:
        keywords: 关键词列表 [(word, frequency), ...]
    """
    if not report or len(report.strip()) == 0:
        return []

    # 使用CountVectorizer提取关键词
    vectorizer = CountVectorizer(stop_words='english', max_features=100)

    try:
        X = vectorizer.fit_transform([report])
        features = vectorizer.get_feature_names_out()
        frequencies = np.array(X.sum(axis=0)).flatten()

        # 获取最重要的关键词
        if len(frequencies) > 0:
            top_indices = np.argsort(frequencies)[-n_keywords:][::-1]
            keywords = [(features[i], int(frequencies[i])) for i in top_indices]
        else:
            keywords = []
    except:
        # 如果处理失败，返回空列表
        keywords = []

    return keywords

def get_keywords(reports):
    """
        获取所有报告对应的关键词
    """
    report_keywords = []
    for i in reports:
        keywords = extract_keywords_from_single_report(i)
        report_keywords.append(keywords)
    return report_keywords

def extract_keywords_from_reports(reports, labels, n_keywords=10):
    """
    整合所有报告中的关键词

    参数:
        reports: 报告文本列表
        labels: 对应标签列表
        n_keywords: 每个类别提取的关键词数量
    """
    # 按类别分离报告
    pneumonia_reports = [report for report, label in zip(reports, labels) if label == 1]
    normal_reports = [report for report, label in zip(reports, labels) if label == 0]

    # 提取关键词
    vectorizer = CountVectorizer(stop_words='english', max_features=1000)

    # 肺炎报告关键词
    if pneumonia_reports:
        X_pneumonia = vectorizer.fit_transform(pneumonia_reports)
        features = vectorizer.get_feature_names_out()
        frequencies = np.array(X_pneumonia.sum(axis=0)).flatten()

        # 获取最重要的关键词
        top_indices = np.argsort(frequencies)[-n_keywords:][::-1]
        pneumonia_keywords = [(features[i], frequencies[i]) for i in top_indices]
    else:
        pneumonia_keywords = []

    # 正常报告关键词
    if normal_reports:
        X_normal = vectorizer.fit_transform(normal_reports)
        features = vectorizer.get_feature_names_out()
        frequencies = np.array(X_normal.sum(axis=0)).flatten()

        # 获取最重要的关键词
        top_indices = np.argsort(frequencies)[-n_keywords:][::-1]
        normal_keywords = [(features[i], frequencies[i]) for i in top_indices]
    else:
        normal_keywords = []

    return pneumonia_keywords, normal_keywords


def create_keyword_visualization(pneumonia_keywords, normal_keywords, output_path=None):
    """
    创建关键词可视化

    参数:
        pneumonia_keywords: 肺炎相关关键词列表
        normal_keywords: 正常相关关键词列表
        output_path: 输出路径（可选）
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 肺炎关键词
    if pneumonia_keywords:
        words, frequencies = zip(*pneumonia_keywords)
        axes[0].barh(range(len(words)), frequencies)
        axes[0].set_yticks(range(len(words)))
        axes[0].set_yticklabels(words,fontsize=24)
        axes[0].set_title('肺炎报告关键词',fontsize=24)
        axes[0].invert_yaxis()

    # 正常关键词
    if normal_keywords:
        words, frequencies = zip(*normal_keywords)
        axes[1].barh(range(len(words)), frequencies)
        axes[1].set_yticks(range(len(words)))
        axes[1].set_yticklabels(words,fontsize=24)
        axes[1].set_title('正常报告关键词',fontsize=24)
        axes[1].invert_yaxis()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def create_wordcloud(keywords, output_path=None):
    """
    创建词云图

    参数:
        keywords: 关键词列表（包含频率）
        output_path: 输出路径（可选）
    """
    word_freq = {word: freq for word, freq in keywords}
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white'
    ).generate_from_frequencies(word_freq)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def analyze_text_feature_importance(text_features, labels, feature_names, output_dir):
    """
    分析文本特征重要性

    参数:
        text_features: 文本特征矩阵
        labels: 标签列表
        feature_names: 特征名称列表
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 按类别分离特征
    pneumonia_features = text_features[labels == 1]
    normal_features = text_features[labels == 0]

    # 计算每个特征的平均重要性
    pneumonia_importance = np.mean(pneumonia_features, axis=0)
    normal_importance = np.mean(normal_features, axis=0)

    # 获取最重要的特征
    n_top = min(20, len(feature_names))
    top_pneumonia_idx = np.argsort(pneumonia_importance)[-n_top:][::-1]
    top_normal_idx = np.argsort(normal_importance)[-n_top:][::-1]

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    # 肺炎相关特征
    pneumonia_top_features = [feature_names[i] for i in top_pneumonia_idx]
    pneumonia_top_importance = [pneumonia_importance[i] for i in top_pneumonia_idx]

    axes[0].barh(range(len(pneumonia_top_features)), pneumonia_top_importance)
    axes[0].set_yticks(range(len(pneumonia_top_features)))
    axes[0].set_yticklabels(pneumonia_top_features)
    axes[0].set_title('肺炎相关文本特征重要性')
    axes[0].invert_yaxis()

    # 正常相关特征
    normal_top_features = [feature_names[i] for i in top_normal_idx]
    normal_top_importance = [normal_importance[i] for i in top_normal_idx]

    axes[1].barh(range(len(normal_top_features)), normal_top_importance)
    axes[1].set_yticks(range(len(normal_top_features)))
    axes[1].set_yticklabels(normal_top_features)
    axes[1].set_title('正常相关文本特征重要性')
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'text_feature_importance.png'), bbox_inches='tight', dpi=300)
    plt.close()

    return {
        'pneumonia_features': list(zip(pneumonia_top_features, pneumonia_top_importance)),
        'normal_features': list(zip(normal_top_features, normal_top_importance))
    }