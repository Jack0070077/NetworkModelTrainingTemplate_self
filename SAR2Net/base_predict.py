"""
网络测试模板,由于导入训练好的权重并识别数据集中的图片数据
"""
import torch
from torchvision import transforms
from my_utlis.my_dataset import MyDatasetGenerator
from my_utlis.netTools import predict_label
from my_module.resnet_cifar10_3x32x32 import resnet20, resnet32, resnet44, resnet56

batch_size = 6  # 一次展示的图片个数
net = resnet20()
val_data_path = 'data/cifar10-images-32x32/test'
weight_path = 'logs/resnet20-32x32-npy/logs_20250401_171304/weight_save/net_weight_epoch99_valAcc_0.87840'
text_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']  # CIFAR-10 的标签

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
val_data = MyDatasetGenerator(root_dir=val_data_path, transform=test_transforms)
val_iter = torch.utils.data.DataLoader(val_data, batch_size, shuffle=True)

# 加载预训练的模型权重
net.load_state_dict(torch.load(weight_path, weights_only=True))
net.eval()
if __name__ == '__main__':
    predict_label(net, val_iter, batch_size, text_labels)  # 进行预测并显示结果
