import random
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from PIL import Image
import h5py
import os


class GradCAMpp:
    """Grad-CAM++可视化类"""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []

        # 注册前向和反向钩子
        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # 注册钩子
        forward_handle = self.target_layer.register_forward_hook(forward_hook)
        backward_handle = self.target_layer.register_backward_hook(backward_hook)
        self.hook_handles = [forward_handle, backward_handle]

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def generate_cam(self, input_tensor, target_class=None):
        # 前向传播
        output = self.model(input_tensor)

        # 处理二分类问题
        if output.shape[1] == 1:  # 二分类，输出维度为1
            if target_class is None:
                # 使用0.5作为阈值决定目标类别
                target_class = 1 if output.item() > 0.5 else 0

            # 对于二分类问题，我们使用输出值本身作为目标
            # 如果目标类别是1，我们希望增加输出值；如果是0，我们希望减少输出值
            if target_class == 1:
                target = output
            else:
                target = 1 - output
        else:  # 多分类问题
            if target_class is None:
                target_class = output.argmax(dim=1).item()

            # 创建一个one-hot向量
            one_hot = torch.zeros_like(output)
            one_hot[0, target_class] = 1
            target = one_hot * output

        # 反向传播
        self.model.zero_grad()
        target.backward(retain_graph=True)

        # 计算Grad-CAM++
        gradients = self.gradients[0].cpu().numpy()
        activations = self.activations[0].cpu().numpy()

        # 计算权重
        weights = np.mean(gradients, axis=(1, 2), keepdims=True)

        # 生成热图
        cam = np.sum(weights * activations, axis=0)
        cam = np.maximum(cam, 0)  # ReLU
        cam = cv2.resize(cam, input_tensor.shape[2:][::-1])
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)

        return cam, target_class


def visualize_image_interpretability(model, image_path, transform, target_layer,
                                     report_text=None, output_path=None):
    """
    可视化图像解释性分析

    参数:
        model: 训练好的图像模型
        image_path: 图像路径
        transform: 图像预处理变换
        target_layer: 目标层（用于Grad-CAM++）
        report_text: 对应的文本报告（可选）
        output_path: 输出路径（可选）
    """
    # 加载和预处理图像
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0)

    # 创建Grad-CAM++实例
    cam_generator = GradCAMpp(model, target_layer)

    # 生成热图
    heatmap, pred_class = cam_generator.generate_cam(input_tensor)

    # 转换为彩色热图
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    # 原始图像
    img_np = np.array(img)
    img_np = cv2.resize(img_np, (224, 224))

    # 叠加热图
    overlay = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)

    # 创建可视化布局
    if report_text:
        # 如果有文本信息，创建2行3列的布局
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 第一行：图像和热图
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title('原始图像', fontsize=14)
        axes[0, 0].axis('off')

        axes[0, 1].imshow(heatmap, cmap='jet')
        axes[0, 1].set_title('Grad-CAM++ 热图', fontsize=14)
        axes[0, 1].axis('off')

        axes[0, 2].imshow(overlay)
        axes[0, 2].set_title('叠加热图 (预测: {})'.format('肺炎' if pred_class == 1 else '正常'), fontsize=14)
        axes[0, 2].axis('off')

        # 限制文本长度，避免显示过长
        display_text = report_text[:300] + "..." if len(report_text) > 300 else report_text
        axes[1, 0].text(0.05, 0.85, display_text, fontsize=14, transform=axes[1, 0].transAxes,
                        verticalalignment='top', wrap=True)
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].axis('off')
        axes[1, 0].set_title('文本关键字', fontsize=16)

        axes[1, 1].axis('off')
        axes[1, 2].axis('off')
    else:
        # 如果没有文本信息，保持原来的1行3列布局
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img_np)
        axes[0].set_title('原始图像')
        axes[0].axis('off')

        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM++ 热图')
        axes[1].axis('off')

        axes[2].imshow(overlay)
        axes[2].set_title('叠加热图 (预测: {})'.format('肺炎' if pred_class == 1 else '正常'))
        axes[2].axis('off')

    plt.tight_layout()

    if output_path:
        # 只保存预测是肺炎的
        if pred_class == 1:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

    cam_generator.remove_hooks()

    return heatmap, pred_class


def batch_image_interpretability(model, image_paths, labels, transform, target_layer, output_dir,
                                 reports=None,num_samples=10):
    """
    批量处理图像可解释性分析

    参数:
        model: 训练好的图像模型
        image_paths: 图像路径列表
        labels: 对应标签列表
        transform: 图像预处理变换
        target_layer: 目标层
        output_dir: 输出目录
        reports: 对应的报告文本列表（可选）
        keywords: 关键词元组 (pneumonia_keywords, normal_keywords)（可选）
        num_samples: 总样本数
    """
    os.makedirs(output_dir, exist_ok=True)

    # 选择肺炎样本
    pneumonia_indices = [i for i, label in enumerate(labels) if label == 1]

    # 随机选择样本
    selected_pneumonia = random.sample(pneumonia_indices, min(num_samples, len(pneumonia_indices)))

    results = []
    for idx in selected_pneumonia:
        img_path = image_paths[idx]
        label = labels[idx]

        # 获取对应的报告文本（如果提供）
        report_text = reports[idx] if reports and idx < len(reports) else None

        output_path = os.path.join(output_dir, f"interpretability_{idx}.png")
        heatmap, pred_class = visualize_image_interpretability(
            model, img_path, transform, target_layer, report_text, output_path
        )

        results.append({
            'image_path': img_path,
            'true_label': label,
            'predicted_label': pred_class,
            'heatmap': heatmap,
            'output_path': output_path,
            'index': idx
        })

    return results