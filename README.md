# MIMIC_MULTI_MODEL
基于特征优化与多模态融合的肺炎检测模型

1、MIMIC-CXR数据集获取方式：在physionet官网中申请资格并下载，地址 https://physionet.org/content/mimic-cxr-jpg/2.1.0/

2、运行image_classify.py与text_classify.py，进行单模态分类，并得到预训练模型

3、运行extract_image_features.py与extract_text_features.py得到图像和文本优化特征

4、运行res_cross_fusion.py进行多模态融合训练

5、运行interpretability.py进行可解释分析并可视化

##############################################################################################################################################

Pneumonia detection model based on feature optimization and multimodal fusion

1. MIMIC-CXR datasets access: application in physionet website and download, at https://physionet.org/content/mimic-cxr-jpg/2.1.0/

2. Run image_classify.py and text_classify.py to perform single-modal classification and obtain the pre-trained model

3. Run extract_image_features.py and extract_text_features.py to obtain the optimized features of the image and text

4. Run res_cross_fusion.py for multimodal fusion training

5. Run interpretability.py for interpretability analysis and visualization
