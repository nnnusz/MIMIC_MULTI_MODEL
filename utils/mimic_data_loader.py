import os
import random
import pandas as pd
from tqdm import tqdm

random.seed(42)

IMAGE_PATH = "G:/datasets/MIMIC-CXR/MIMIC_IMG"
REPORT_PATH = "G:/datasets/MIMIC-CXR/MIMIC_REPORT"

SAMPLE1 = 1000
SAMPLE0 = 3000

def get_data(denoising=False):
    df = pd.read_csv('../datasets/mimic_pneumonia.csv')
    df_meta = pd.read_csv('../datasets/MIMIC-CXR/mimic-cxr-2.0.0-metadata.csv')
    images_1 = []
    reports_1 = []
    labels_1 = []
    images_0 = []
    reports_0 = []
    labels_0 = []
    ids = list(df['study_id'])
    for i in ids:
        data = df[df['study_id']==i].iloc[0]
        findings = data['findings']
        impression = data['impression']
        if type(findings) != str or type(impression) != str:
            continue
        report = findings + '####' + impression
        label = int(data['Pneumonia'])
        meta_data = df_meta[df_meta['study_id']==i]
        imgs = list(meta_data['dicom_id'])
        if len(imgs) == 0:
            continue
        for img in imgs:
            if denoising:
                img_path = os.path.join(IMAGE_PATH,'denoising',img + '.jpg')
            else:
                img_path = os.path.join(IMAGE_PATH,img +'.jpg')
            if not os.path.exists(img_path):
                continue
            if label == 1:
                images_1.append(img_path)
                reports_1.append(report)
                labels_1.append(label)
            else:
                images_0.append(img_path)
                reports_0.append(report)
                labels_0.append(label)
    #随机抽取
    selected_images = []
    selected_reports = []
    selected_labels = []
    random_idx_1 = random.sample(range(len(images_1)),SAMPLE1)
    for i in random_idx_1:
        selected_images.append(images_1[i])
        selected_reports.append(reports_1[i])
        selected_labels.append(labels_1[i])
    random_idx_0 = random.sample(range(len(images_0)),SAMPLE0)
    for i in random_idx_0:
        selected_images.append(images_0[i])
        selected_reports.append(reports_0[i])
        selected_labels.append(labels_0[i])
    print('样本总数量：',len(selected_images))
    print(f'标签1：{selected_labels.count(1)}')
    print(f'标签0：{selected_labels.count(0)}')
    return selected_images,selected_reports,selected_labels

#获取图像方向
def get_image_angles(images):
    angles = []
    df_meta = pd.read_csv('../datasets/MIMIC-CXR/mimic-cxr-2.0.0-metadata.csv')
    for i in images:
        angle = 'A'
        dicom_id = os.path.basename(i).split('.')[0]
        df = df_meta.loc[df_meta['dicom_id']==dicom_id]
        if len(df) > 0:
            pos = df.iloc[0]['ViewPosition']
            if type(pos) == str and 'L' in pos:
                angle = 'L'
        angles.append(angle)
    return angles