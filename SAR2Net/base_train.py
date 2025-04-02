"""
小规模网络训练模板,使用打包成npy文件的数据集加速训练
1.先读取图片数据集,调用generate_NPY_files函数打包成npy文件,加速dataloader的读取
2.在DataLoader中调用transforms进行数据增强
"""
import torch
import os
import numpy as np
import datetime
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from my_utlis.netTools import Animator111, init_weight, train, save_info
from my_utlis.my_dataset import generate_NPY_files, my_collate_fn
from my_module.resnet_cifar10_3x32x32 import resnet20, resnet32, resnet44, resnet56

# -------------------------------------------- Params Setting -------------------------------------------------- #
batch_size = 64
lr = 7e-4
l2_alpha = 5e-2
EPOCH, start_epoch, is_pretrained = 100, 0, False
net = resnet20()
log_name = 'resnet20-32x32-npy'
data_path = './data/cifar10-images-32x32'
weight_dir = ''

train_transforms = transforms.Compose([  # 训练数据增强
    # transforms.ToTensor(),  # 转化为tensor类型
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomHorizontalFlip(),  # 随机水平镜像
    transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)),  # 随机遮挡
    transforms.RandomCrop(size=32, padding=4),  # 随机裁剪
    transforms.RandomRotation(15),  # 随机旋转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
])
test_transforms = transforms.Compose([  # 测试数据增强
    # transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# optim_fn = torch.optim.SGD(net.parameters(), lr, weight_decay=l2_alpha, momentum=0.9)  # 随机梯度下降
optim_fn = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=l2_alpha, betas=(0.9, 0.999))  # AdamW 优化器

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim_fn, T_max=EPOCH, eta_min=0.001 * lr, last_epoch=start_epoch - 1)  # 余弦退火
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optim_fn, milestones=[45, 60, 80], last_epoch=start_epoch - 1)# 里程碑衰减
# scheduler = None
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)  # 交叉熵

# -------------------------------------------------------------------------------------------------------------- #

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
print()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if is_pretrained:
    print(f'load weights from:{weight_dir}')
    net.load_state_dict(torch.load(weight_dir, map_location=device))
else:
    net.apply(init_weight)
os.makedirs(
    name=os.path.join('logs', log_name, f'logs_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'), exist_ok=True)
logs_root = os.path.join('logs', log_name, f'logs_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}')
os.makedirs(name=os.path.join(logs_root, 'log_files'), exist_ok=True)
writer = SummaryWriter(log_dir=os.path.join(logs_root, 'log_files'))

data_npy_path = os.path.join(data_path, 'npy_files')
train_features_path = os.path.join(data_npy_path, 'train_features.npy')
train_targets_path = os.path.join(data_npy_path, 'train_targets.npy')
val_features_path = os.path.join(data_npy_path, 'val_features.npy')
val_targets_path = os.path.join(data_npy_path, 'val_targets.npy')

animator = Animator111(xlabel='epoch', xlim=[0, EPOCH], nrows=1, ncols=2, figsize=(5.5, 3.7),
                       legend=[['train_loss', 'val_loss'], ['train_acc', 'val_acc']],
                       save_dir=os.path.join(logs_root, 'fig_save'), fmts=['-o'], markersizes=[4, 4, 4, 4])
print(f'# ------------------------------------------------------------------------------ #\n'
      f'#        device choose: {device}\n'
      f'#        net={net.__class__}\n'
      f'#        start_epoch={start_epoch},EPOCH={EPOCH}\n'
      f'#        optim_fn={optim_fn.__class__}\n'
      f'#        scheduler_fn={scheduler.__class__}\n'
      f'#        loss_fn={loss_fn.__class__}\n'
      f'# ------------------------------------------------------------------------------ #')

if __name__ == '__main__':
    # 1.判断是否预训练
    if (os.path.exists(train_features_path)
            and os.path.exists(train_targets_path)
            and os.path.exists(val_features_path)
            and os.path.exists(val_targets_path)):
        print('.npy files are exist, load and train')
    else:
        print('.npy files NOT exist, generate .npy files...')
        generate_NPY_files(data_path, data_npy_path)
    # 2.数据集打包
    train_features = torch.tensor(data=np.load(train_features_path), dtype=torch.float32)
    train_targets = torch.tensor(data=np.load(train_targets_path), dtype=torch.long)
    val_features = torch.tensor(data=np.load(val_features_path), dtype=torch.float32)
    val_targets = torch.tensor(data=np.load(val_targets_path), dtype=torch.long)
    train_dataset = torch.utils.data.TensorDataset(train_features, train_targets)
    val_dataset = torch.utils.data.TensorDataset(val_features, val_targets)
    train_iter = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True,
        collate_fn=lambda batch: my_collate_fn(batch, train_transforms),
    )
    val_iter = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True,
        collate_fn=lambda batch: my_collate_fn(batch, test_transforms)
    )
    # 3.导入gpu
    loss_fn = loss_fn.to(device)
    net.to(device)
    # 4.保存本轮参数信息
    save_info(logs_root=logs_root, batch_size=batch_size, lr=lr, l2_alpha=l2_alpha,
              patience=None, is_pretrained=is_pretrained, weight_dir=weight_dir,
              strat_epoch=start_epoch, end_epoch=start_epoch + EPOCH - 1,
              train_trans=train_transforms, val_trans=test_transforms,
              optim_fn=optim_fn, lr_schedule=scheduler, loss_fn=loss_fn, net=net)
    # 5.网络训练
    train(logs_root=logs_root, device=device, start_epoch=start_epoch, EPOCH=EPOCH,
          data_iter=[train_iter, val_iter], net=net, optim_fn=optim_fn, loss_fn=loss_fn, scheduler=scheduler,
          animator=animator, writer=writer, is_earlyStop=False)
