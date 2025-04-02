"""
将数据集文件转换成图片并保存
"""
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import tqdm
import os

# 第一步：将 CIFAR - 10 转换为 JPG 图像
root_dir = '../data/cifar10-images-32x32'
if not os.path.exists(root_dir): os.makedirs(root_dir)
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
trainset = torchvision.datasets.CIFAR10(root='../data/cifar10', train=True,
                                        download=True, transform=transforms.ToTensor())
testset = torchvision.datasets.CIFAR10(root='../data/cifar10', train=False,
                                       download=True, transform=transforms.ToTensor())

target_size = (32, 32)# 指定图片大小


def save_images(dataset, dataset_type):
    h = tqdm.tqdm(total=len(dataset))
    for i in range(len(dataset)):
        image, label = dataset[i]
        image = transforms.ToPILImage()(image)
        # 调整图片大小
        image = image.resize(target_size, Image.BICUBIC)
        class_name = classes[label]
        class_dir = os.path.join(root_dir, dataset_type, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        image_path = os.path.join(class_dir, f'{i}.jpg')
        image.save(image_path)
        h.update(1)

save_images(trainset, 'train')
save_images(testset, 'test')

print("CIFAR - 10 数据集已成功转换为指定大小的 JPG 图像")
