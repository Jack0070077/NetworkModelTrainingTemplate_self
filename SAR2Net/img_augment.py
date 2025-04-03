"""
1.数据集增强,扩充
2.数据分割
3.写入npy文件加速训练
"""
import os
import random
import cv2
import math
import albumentations as A
import splitfolders
import tqdm
from my_utlis.netTools import generate_npy_from_tiff

original_data_root = 'data/original_data'
augment_data_root = 'data/augment_data'
size_rate = 15  # 增强后样本数量是最大类别样本数量倍数
dsize = (128, 128)  # 统一缩放大小
class_name_idx = {f'{name}': idx for idx, name in enumerate(sorted(os.listdir(original_data_root)))}  # 类别名称及索引
class_data_num = {f'{name}': idx for idx, name in enumerate(sorted(os.listdir(original_data_root)))}  # 原始数据集每个类别的样本数量
sample_num_max = 0  # 类别的最大样本数量
for name, _ in class_name_idx.items():
    imgs_num = len(os.listdir(os.path.join(original_data_root, name)))
    class_data_num[name] = imgs_num
    if imgs_num >= sample_num_max:
        sample_num_max = imgs_num

with open(os.path.join('data', 'class_name_idx.txt'), 'w') as f:
    f.write(f'class_names:\n'
            f'{class_name_idx.keys()}\n'
            f'class_names_idx:\n'
            f'{class_name_idx}\n')

sample_num_final = size_rate * sample_num_max  # 所有类别最终的样本数量

each_size_rate = {}  # 每个类别要扩充的比率,即单张图片处理次数(注意小数部分,无法作为一个整的循环,可用作随机数的阈值)
for name, _ in class_name_idx.items():
    each_size_rate[name] = sample_num_final / class_data_num[name]

IMG_transform = A.Compose([
    # 1.旋转 & 翻转（保持空间特性）
    A.Rotate(limit=180, p=0.7),  # SAR 图像无方向性，可以任意旋转
    A.HorizontalFlip(p=0.5),  # 水平翻转
    A.VerticalFlip(p=0.5),  # 垂直翻转

    # 2.模糊 & 噪声（增强抗干扰能力）
    # A.MotionBlur(blur_limit=5, p=0.5),  # 运动模糊，模拟动态场景
    # A.GaussianBlur(blur_limit=(3, 7), p=0.4),  # 高斯模糊，模拟不同焦距
    # A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),  # 高斯噪声，增强鲁棒性
    A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.5),  # 乘性噪声，适用于 SAR

    # 3.对比度调整（增强图像细节）
    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),  # 添加锐化效果
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.3, p=0.7),  # 增强亮度 & 对比度
    # A.CLAHE(clip_limit=4.0, p=0.5),  # 自适应直方图均衡化

    # 4.形变（增加 SAR 目标变化）
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),  # 网格扭曲
    A.ElasticTransform(alpha=1, sigma=50, p=0.5),  # 仿射变换
])

# 1.数据增强,扩充
if True:
    for name, _ in class_name_idx.items():
        save_root = os.path.join(augment_data_root, name)
        os.makedirs(save_root, exist_ok=True)

        imgs_name_list = os.listdir(os.path.join(original_data_root, name))
        imgs_num = len(imgs_name_list)
        imgs_output_root = os.path.join(augment_data_root, name)

        prob = each_size_rate[name] - math.floor(each_size_rate[name])  # 小数部分,作为概率阈值

        img_counter = 0
        h = tqdm.tqdm(total=imgs_num)
        for img_name in imgs_name_list:  # 对一张图像增强
            aug_counter = 0
            img_path = os.path.join(original_data_root, name, img_name)
            img = cv2.imread(img_path)
            # 1.按整数增强
            if math.floor(each_size_rate[name]) >= 1:
                for _ in range(math.floor(each_size_rate[name])):
                    transformed_img = IMG_transform(image=cv2.resize(img, dsize))['image']
                    resized_img = cv2.resize(transformed_img, dsize)
                    aug_filename = f'{name}_img{img_counter}_aug{aug_counter}.tiff'
                    cv2.imwrite(os.path.join(save_root, aug_filename), resized_img)
                    aug_counter += 1
            # 2.按小数概率增强
            if random.uniform(0, 1) <= prob:
                transformed_img = IMG_transform(image=cv2.resize(img, dsize))['image']
                resized_img = cv2.resize(transformed_img, dsize)
                aug_filename = f'{name}_img{img_counter}_aug{aug_counter}.tiff'
                cv2.imwrite(os.path.join(save_root, aug_filename), resized_img)
                aug_counter += 1

            img_counter += 1
            h.update(1)
            h.set_description(f'current_aug_class:{name}')

        h.close()

print('\n> img augment finished.')

# 2.数据分割
split_root = os.path.join('data', 'split_data')
splitfolders.ratio(augment_data_root, output=split_root, seed=1337, ratio=(0.8, 0.2))
print('> img split finished.')
# 3.导出 npy 文件
generate_npy_from_tiff(data_path=split_root,data_npy_path=os.path.join(split_root,'npy_files'))
