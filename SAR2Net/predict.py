"""
网络测试模板,由于导入训练好的权重并识别数据集中的图片数据
"""
import torch
from torchvision import transforms
from my_utlis.netTools import MyDatasetGenerator
from my_utlis.netTools import predict_label_col,predict_label
from my_modules.ResNet_1x128x128 import resnet20, resnet32, resnet44, resnet56

batch_size = 18  # 一次展示的图片个数
net = resnet20()
val_data_path = 'data/split_data/val'  # data/fashion-mnist-img1x128x128/test
weight_path = 'logs/resnet-6classes/logs_20250403_164109/weight_save/Best_val_acc_weights'
# text_labels = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
text_labels = ['BulkCarrier', 'CargoShip', 'ContainerShip', 'CornerReflector', 'Fishing', 'Tanker']
text_labels = sorted(text_labels)
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(),
    #transforms.Normalize(mean=[0.5], std=[0.5])
])
val_data = MyDatasetGenerator(root_dir=val_data_path, transform=test_transforms)
val_iter = torch.utils.data.DataLoader(val_data, batch_size, shuffle=True)
net.load_state_dict(torch.load(weight_path, weights_only=True))
net.eval()
if __name__ == '__main__':
    # predict_label(net, val_iter, num_rows=2, num_cols=6, text_labels=text_labels,scale=2.0)  # 进行预测并显示结果
    predict_label_col(net, val_iter, num_rows=3, num_cols=6, text_labels=text_labels)
