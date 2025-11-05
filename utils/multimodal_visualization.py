import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import os
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def visualize_feature_space(image_features, text_features, labels, output_path=None):
    """可视化特征空间"""
    # 使用t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)

    # 图像特征降维
    image_2d = tsne.fit_transform(image_features)

    # 文本特征降维
    text_2d = tsne.fit_transform(text_features)

    # 创建子图
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 绘制图像特征
    scatter1 = axes[0].scatter(
        image_2d[labels == 0, 0], image_2d[labels == 0, 1],
        c='blue', label='正常', alpha=0.6
    )
    scatter2 = axes[0].scatter(
        image_2d[labels == 1, 0], image_2d[labels == 1, 1],
        c='red', label='肺炎', alpha=0.6
    )
    axes[0].set_title('图像特征空间 (t-SNE)')
    axes[0].legend()

    # 绘制文本特征
    scatter3 = axes[1].scatter(
        text_2d[labels == 0, 0], text_2d[labels == 0, 1],
        c='blue', label='正常', alpha=0.6
    )
    scatter4 = axes[1].scatter(
        text_2d[labels == 1, 0], text_2d[labels == 1, 1],
        c='red', label='肺炎', alpha=0.6
    )
    axes[1].set_title('文本特征空间 (t-SNE)')
    axes[1].legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

    return image_2d, text_2d


def visualize_cross_attention(attention_weights, image_regions, text_tokens, output_path=None):
    """可视化交叉注意力权重"""
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        attention_weights,
        xticklabels=text_tokens,
        yticklabels=image_regions,
        cmap="YlOrRd",
        annot=False,  # 避免过多的注释
        fmt=".2f",
        cbar_kws={'label': '注意力权重'}
    )
    plt.title('图像-文本交叉注意力权重')
    plt.xlabel('文本标记')
    plt.ylabel('图像区域')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def visualize_performance_metrics(performance_metrics, output_path=None):
    """可视化性能指标"""
    metrics = ['准确率', '精确率', '召回率', 'F1分数']
    values = [
        performance_metrics['accuracy'],
        performance_metrics['precision'],
        performance_metrics['recall'],
        performance_metrics['f1']
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('分数')
    ax.set_title('模型性能指标')

    # 在条形上添加数值标签
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{value:.4f}',
                ha='center', va='bottom')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def visualize_confidence_distribution(probs, labels, output_path=None):
    """可视化预测置信度分布"""
    # 提取正类（肺炎）的概率
    pneumonia_probs = probs[labels == 1, 1]  # 第二列是肺炎的概率
    normal_probs = probs[labels == 0, 0]  # 第一列是正常的概率

    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制直方图
    ax.hist(pneumonia_probs, bins=20, alpha=0.7, label='肺炎样本', color='red')
    ax.hist(normal_probs, bins=20, alpha=0.7, label='正常样本', color='blue')

    ax.set_xlabel('预测置信度')
    ax.set_ylabel('样本数量')
    ax.set_title('预测置信度分布')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def create_interactive_dashboard(image_results, text_results, multimodal_results, output_path):
    """创建交互式可视化大屏"""
    # 创建子图
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            '图像Grad-CAM++可视化',
            '文本关键词分析',
            '特征空间分布',
            '交叉注意力权重',
            '模型性能指标',
            '预测置信度分布'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}],
            [{"type": "heatmap"}, {"type": "bar"}, {"type": "histogram"}]
        ]
    )

    # 添加特征空间分布
    image_2d = multimodal_results['image_2d']
    text_2d = multimodal_results['text_2d']
    labels = multimodal_results['labels']

    # 图像特征空间
    fig.add_trace(
        go.Scatter(
            x=image_2d[labels == 0, 0],
            y=image_2d[labels == 0, 1],
            mode='markers',
            name='正常(图像)',
            marker=dict(color='blue')
        ),
        row=1, col=3
    )

    fig.add_trace(
        go.Scatter(
            x=image_2d[labels == 1, 0],
            y=image_2d[labels == 1, 1],
            mode='markers',
            name='肺炎(图像)',
            marker=dict(color='red')
        ),
        row=1, col=3
    )

    # 文本特征空间
    fig.add_trace(
        go.Scatter(
            x=text_2d[labels == 0, 0],
            y=text_2d[labels == 0, 1],
            mode='markers',
            name='正常(文本)',
            marker=dict(color='lightblue')
        ),
        row=1, col=3
    )

    fig.add_trace(
        go.Scatter(
            x=text_2d[labels == 1, 0],
            y=text_2d[labels == 1, 1],
            mode='markers',
            name='肺炎(文本)',
            marker=dict(color='pink')
        ),
        row=1, col=3
    )

    # 添加文本关键词可视化
    pneumonia_keywords, normal_keywords = text_results['keywords']

    fig.add_trace(
        go.Bar(
            y=[kw[0] for kw in pneumonia_keywords],
            x=[kw[1] for kw in pneumonia_keywords],
            orientation='h',
            name='肺炎关键词',
            marker_color='red'
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Bar(
            y=[kw[0] for kw in normal_keywords],
            x=[kw[1] for kw in normal_keywords],
            orientation='h',
            name='正常关键词',
            marker_color='blue'
        ),
        row=1, col=2
    )

    # 添加交叉注意力热图
    attention_weights = multimodal_results['attention_weights']
    fig.add_trace(
        go.Heatmap(
            z=attention_weights,
            colorscale='YlOrRd',
            colorbar=dict(title="注意力权重")
        ),
        row=2, col=1
    )

    # 添加性能指标
    metrics = ['准确率', '精确率', '召回率', 'F1分数']
    values = [
        multimodal_results['performance']['accuracy'],
        multimodal_results['performance']['precision'],
        multimodal_results['performance']['recall'],
        multimodal_results['performance']['f1']
    ]

    fig.add_trace(
        go.Bar(
            x=metrics,
            y=values,
            marker_color=['blue', 'green', 'orange', 'red'],
            name='性能指标'
        ),
        row=2, col=2
    )

    # 添加置信度分布
    probs = multimodal_results['probs']
    labels = multimodal_results['labels']

    # 肺炎样本的置信度
    pneumonia_probs = probs[labels == 1, 1]
    # 正常样本的置信度
    normal_probs = probs[labels == 0, 0]

    fig.add_trace(
        go.Histogram(
            x=pneumonia_probs,
            name='肺炎样本',
            marker_color='red',
            opacity=0.7,
            nbinsx=20
        ),
        row=2, col=3
    )

    fig.add_trace(
        go.Histogram(
            x=normal_probs,
            name='正常样本',
            marker_color='blue',
            opacity=0.7,
            nbinsx=20
        ),
        row=2, col=3
    )

    # 更新布局
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="多模态肺炎检测可解释性分析大屏"
    )

    # 保存为HTML
    fig.write_html(output_path)


def generate_interpretability_report(image_results, text_results, multimodal_results, output_path):
    """生成可解释性分析报告"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 多模态肺炎检测模型可解释性分析报告\n\n")

        f.write("## 1. 图像特征分析\n\n")
        f.write("通过对图像特征的Grad-CAM++分析，模型主要关注以下区域进行肺炎检测:\n\n")

        for i, result in enumerate(image_results[:5]):  # 只显示前5个结果
            f.write(f"### 样本 {i + 1}\n")
            f.write(f"- 图像路径: {result['image_path']}\n")
            f.write(f"- 真实标签: {'肺炎' if result['true_label'] == 1 else '正常'}\n")
            f.write(f"- 预测标签: {'肺炎' if result['predicted_label'] == 1 else '正常'}\n")
            f.write(f"- 预测正确: {'是' if result['true_label'] == result['predicted_label'] else '否'}\n")
            f.write(f"- 热图保存路径: {result['output_path']}\n\n")

        f.write("## 2. 文本特征分析\n\n")
        f.write("### 肺炎报告关键词\n")
        for word, freq in text_results['keywords'][0]:
            f.write(f"- {word}: {freq}\n")

        f.write("\n### 正常报告关键词\n")
        for word, freq in text_results['keywords'][1]:
            f.write(f"- {word}: {freq}\n")

        f.write("\n## 3. 多模态融合分析\n\n")
        f.write("通过交叉注意力机制，模型能够建立图像区域与文本关键词之间的关联:\n\n")

        f.write("### 模型性能指标\n")
        f.write(f"- 准确率: {multimodal_results['performance']['accuracy']:.4f}\n")
        f.write(f"- 精确率: {multimodal_results['performance']['precision']:.4f}\n")
        f.write(f"- 召回率: {multimodal_results['performance']['recall']:.4f}\n")
        f.write(f"- F1分数: {multimodal_results['performance']['f1']:.4f}\n\n")

        f.write("### 最重要的图像-文本关联\n")
        attention_weights = multimodal_results['attention_weights']
        if attention_weights is not None:
            # 找出最相关的图像-文本对
            max_indices = np.unravel_index(np.argmax(attention_weights), attention_weights.shape)
            f.write(
                f"- 最强关联: 图像区域 {max_indices[0]} ↔ 文本标记 {max_indices[1]} (权重: {attention_weights[max_indices]:.4f})\n\n")

        f.write("## 4. 结论\n\n")
        f.write("本可解释性分析表明:\n")
        f.write("1. 图像模型主要关注肺野区域进行肺炎检测\n")
        f.write("2. 文本模型中，'consolidation', 'opacity'等关键词与肺炎高度相关\n")
        f.write("3. 多模态融合通过交叉注意力机制有效结合了图像和文本信息\n")
        f.write("4. 模型在测试集上表现良好，准确率达到{:.2f}%\n".format(multimodal_results['performance']['accuracy'] * 100))