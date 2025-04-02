"""
读取文件夹中的图片文件，并生成dataset，供dataloader生成data_iter
暂时只能打开jpg格式文件
"""
import os
import torch
import torchvision.transforms
from PIL import Image
import numpy as np


class MyDatasetGenerator(torch.utils.data.Dataset):  # 生成dataset的类,读取的是离散的数据集
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir  # 根目录
        self.labels_dir = sorted(os.listdir(self.root_dir))  # 数据集下的类别文件夹列表
        self.labels_index = {label: index for index, label in enumerate(self.labels_dir)}  # 标签对应的标号字典
        self.transform = transform  # 应用的变换
        self.imgs_dir, self.imgs_label = [], []  # 所有的图片路径列表和对应的标签标号列表
        for label in self.labels_dir:
            imgs_name = os.listdir(os.path.join(self.root_dir, label))  # 图片路径
            for img_name in imgs_name:
                img_path = os.path.join(self.root_dir, label, img_name)
                self.imgs_dir.append(img_path)
                self.imgs_label.append(self.labels_index[label])

    def __getitem__(self, item):
        img_path = self.imgs_dir[item]
        img_label = self.imgs_label[item]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, img_label

    def __len__(self):
        return len(self.imgs_dir)


def my_collate_fn(batch, transform):  # 将一个batch堆叠成一个批次进行transforms,使用lambda表达式调用
    images = [item[0] for item in batch]  # batch是一个列表，每个元素是一个元组(img, label)
    labels = [item[1] for item in batch]
    images = torch.stack([transform(img) for img in images])  # 对每个图片应用transform
    labels = torch.tensor(labels)
    return images, labels


def generate_NPY_files(data_path, data_npy_path):  # 将数据转为Tensor导入内存,加速训练
    # !!!!!   不能在这里面进行预处理，否则与出事的数据集没有区别，还是会过拟合   !!!!!
    train_path = os.path.join(data_path, 'train')
    val_path = os.path.join(data_path, 'test')
    os.makedirs(data_npy_path, exist_ok=True)
    train_set = MyDatasetGenerator(train_path, torchvision.transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), shuffle=True)
    for data in train_loader:
        train_features, train_targets = data
        train_features = train_features.numpy()
        train_targets = train_targets.numpy()
    os.makedirs(name=os.path.join(data_npy_path), exist_ok=True)
    np.save(os.path.join(data_npy_path, 'train_features'), train_features)
    np.save(os.path.join(data_npy_path, 'train_targets'), train_targets)
    print('train_date.npy has finished generate')
    val_set = MyDatasetGenerator(val_path, torchvision.transforms.ToTensor())
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=len(val_set), shuffle=True)
    for data in val_loader:
        val_features, val_targets = data
        val_features = val_features.numpy()
        val_targets = val_targets.numpy()
    np.save(os.path.join(data_npy_path, 'val_features'), val_features)
    np.save(os.path.join(data_npy_path, 'val_targets'), val_targets)
    print('val_date.npy has finished generate')
