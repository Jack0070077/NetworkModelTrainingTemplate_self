"""
常用的方法放在这里
"""
import os
import torch
from torch import nn
import torchvision
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class Animator:  # 动态绘制训练过程中的损失和准确率曲线
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'),
                 nrows=1, ncols=1, figsize=(10, 4), save_dir='add_FigSave'):
        import matplotlib.pyplot as plt
        if legend is None: legend = [[] for _ in range(nrows * ncols)]
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1: self.axes = [self.axes, ]
        self.config_axes = lambda: [
            plt.setp(ax, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, xscale=xscale, yscale=yscale) for ax in self.axes
        ]
        self.X = [[] for _ in range(nrows * ncols)]
        self.Y = [[] for _ in range(nrows * ncols)]
        self.fmts, self.legend = fmts, legend
        self.nrows, self.ncols = nrows, ncols
        self.ylim = ylim if ylim is not None else [None] * (nrows * ncols)
        self.save_dir = save_dir  # 保存图片的目录
        os.makedirs(self.save_dir, exist_ok=True)  # 确保保存目录存在

    def add(self, epoch, y, subplot_index=0):  # 将数据添加到指定的子图中
        if subplot_index < 0 or subplot_index >= self.nrows * self.ncols:  # 确保 subplot_index 在有效范围内
            raise ValueError(f"subplot_index must be between 0 and {self.nrows * self.ncols - 1}")
        self.X[subplot_index].append(epoch)  # 添加数据到指定的子图
        if len(self.Y[subplot_index]) == 0:
            self.Y[subplot_index] = [[y]]
        else:
            self.Y[subplot_index][0].append(y)
        self.axes[subplot_index].cla()  # 清除指定子图的内容并重新绘制
        self.axes[subplot_index].grid(True)  # 启用网格线
        # if self.ylim[subplot_index] :
        #     self.axes[subplot_index].set_ylim(self.ylim[subplot_index])
        for y, fmt in zip(self.Y[subplot_index], self.fmts):
            self.axes[subplot_index].plot(self.X[subplot_index], y, fmt)
        self.config_axes()
        self.axes[subplot_index].legend(self.legend[subplot_index])  # 使用对应的图例
        plt.pause(0.05)  # 暂停一下，让图像更新
        if (epoch + 1) % 5 == 0:
            self.savefig(epoch)  # 保存图像

    def savefig(self, epoch):  # 保存图像
        filename = os.path.join(self.save_dir, f"epoch_{epoch}.png")
        self.fig.savefig(filename)
        # print(f"图像已保存为 {filename}")


class Animator111:  # 动态绘制训练过程中的损失和准确率曲线,包含marker_size设置
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'),
                 markersizes=None, nrows=1, ncols=1, figsize=(7, 5), save_dir='add_FigSave'):
        import matplotlib.pyplot as plt
        if legend is None: legend = [[] for _ in range(nrows * ncols)]
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1: self.axes = [self.axes, ]
        self.config_axes = lambda: [
            plt.setp(ax, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, xscale=xscale, yscale=yscale) for ax in self.axes
        ]
        self.X = [[] for _ in range(nrows * ncols)]
        self.Y = [[] for _ in range(nrows * ncols)]
        self.fmts, self.legend = fmts, legend
        self.nrows, self.ncols = nrows, ncols
        self.ylim = ylim if ylim is not None else [None] * (nrows * ncols)
        self.save_dir = save_dir  # 保存图片的目录
        os.makedirs(self.save_dir, exist_ok=True)  # 确保保存目录存在
        self.markersizes = markersizes if markersizes is not None else [6] * len(fmts)

    def add(self, epoch, y, subplot_index=0):  # 将数据添加到指定的子图中
        if subplot_index < 0 or subplot_index >= self.nrows * self.ncols:  # 确保 subplot_index 在有效范围内
            raise ValueError(f"subplot_index must be between 0 and {self.nrows * self.ncols - 1}")
        self.X[subplot_index].append(epoch)  # 添加数据到指定的子图
        if len(self.Y[subplot_index]) == 0:
            self.Y[subplot_index] = [[y]]
        else:
            self.Y[subplot_index][0].append(y)
        self.axes[subplot_index].cla()  # 清除指定子图的内容并重新绘制
        self.axes[subplot_index].grid(True)  # 启用网格线
        for y, fmt, markersize in zip(self.Y[subplot_index], self.fmts, self.markersizes):
            self.axes[subplot_index].plot(self.X[subplot_index], y, fmt, markersize=markersize)
        self.config_axes()
        self.axes[subplot_index].legend(self.legend[subplot_index])  # 使用对应的图例
        plt.pause(0.05)  # 暂停一下，让图像更新
        if (epoch + 1) % 5 == 0:
            self.savefig(epoch)  # 保存图像

    def savefig(self, epoch):  # 保存图像
        filename = os.path.join(self.save_dir, f"epoch_{epoch}.png")
        self.fig.savefig(filename)
        # print(f"图像已保存为 {filename}")


def init_weight(m):  # 初始化全连接层的权重
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        # torch.nn.init.kaiming_uniform_(m.weight)


def set_seed(seed):  # 固定随机数种子,似乎不起作用,笑死
    random.seed(seed)  # Python随机数种子
    np.random.seed(seed)  # Numpy随机数种子
    torch.manual_seed(seed)  # PyTorch CPU随机数种子
    torch.cuda.manual_seed(seed)  # PyTorch CUDA随机数种子
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU，设置所有GPU的随机数种子
    torch.backends.cudnn.deterministic = True  # 保证CUDNN的确定性
    torch.backends.cudnn.benchmark = False  # 禁用CUDNN自动优化，使计算更稳定


def train(logs_root, device, start_epoch, EPOCH,
          data_iter: list, net, optim_fn, loss_fn, scheduler=None,
          animator=None, writer=None, is_earlyStop=False, patience=10):  # 网络训练函数,包含训练和测试
    import tqdm
    h = tqdm.tqdm(total=EPOCH)
    train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []
    best_val_acc, counter = 0.0, 0
    train_iter, val_iter = data_iter[0], data_iter[1]
    for epoch in range(start_epoch, start_epoch + EPOCH):  # 一次epoch训练
        train_loss, train_acc, val_loss, val_acc = 0., 0., 0., 0.
        net.train()  # 训练模式
        for feature, label in train_iter:
            feature, label = feature.to(device), label.to(device)  # 打到gpu
            net.zero_grad()  # 清空上一轮梯度
            label_hat = net(feature)  # 网络输出
            loss = loss_fn(label_hat, label)  # 计算损失
            loss.mean().backward()  # 反向传播
            optim_fn.step()  # 更新权重
            train_loss += loss.item()
            train_acc += (torch.argmax(label_hat, dim=1) == label).sum().item()  # 计算准确率
        train_loss /= len(train_iter.dataset)
        train_acc /= len(train_iter.dataset)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        net.eval()  # 测试模式
        with torch.no_grad():
            for feature, label in val_iter:
                feature, label = feature.to(device), label.to(device)
                label_hat = net(feature)
                loss = loss_fn(label_hat, label)
                val_loss += loss.item()
                val_acc += (torch.argmax(label_hat, dim=1) == label).sum().item()
            val_loss /= len(val_iter.dataset)
            val_acc /= len(val_iter.dataset)
            val_acc_list.append(val_acc)
            val_loss_list.append(val_loss)

        if scheduler is not None:  # 更新学习率
            scheduler.step()
        if val_acc > best_val_acc:  # 保存最好一轮的val-acc对应的权重
            best_val_acc = val_acc
            weight_dir = os.path.join(logs_root, 'weight_save')
            os.makedirs(weight_dir, exist_ok=True)
            torch.save(net.state_dict(), os.path.join(weight_dir, f'Best_val_acc_weights'))
            with open(os.path.join(logs_root, 'best_val_acc.txt'), 'w') as f:
                f.write(f'best val_acc = {best_val_acc}\n'
                        f'epoch = {epoch}\n'
                        f'')

        # if is_earlyStop:  # 早停
        #     if val_acc > best_val_acc:
        #         best_val_acc = val_acc
        #         counter = 0
        #     else:
        #         counter += 1
        #     if counter >= patience:
        #         print("> Early stopping triggered.")
        #         break

        if animator is not None:  # 动态绘制曲线
            animator.add(epoch, (train_loss, val_loss), subplot_index=0)  # 在第一个子图绘制损失
            animator.add(epoch, (train_acc, val_acc), subplot_index=1)  # 在第二个子图绘制准确率
        if writer is not None:  # TensorBoard记录
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_acc', train_acc, epoch)
            writer.add_scalar('val_loss', val_loss, epoch)
            writer.add_scalar('val_acc', val_acc, epoch)

        if (epoch + 1) % 10 == 0:  # 保存权重文件
            weight_dir = os.path.join(logs_root, 'weight_save')
            os.makedirs(weight_dir, exist_ok=True)
            torch.save(net.state_dict(), os.path.join(weight_dir, f'net_weight_epoch{epoch}_valAcc_{val_acc:.5f}'))

        h.update(1)
        h.set_description(
            f">> epoch={epoch},train_acc={train_acc * 100:.2f}%,val_acc={val_acc * 100:.2f}%,lr={optim_fn.param_groups[0]['lr']:.10f}")

    writer.close()
    plt.show()


def save_info(logs_root, batch_size, lr, l2_alpha, patience, is_pretrained, weight_dir, strat_epoch, end_epoch,
              train_trans, val_trans, optim_fn, lr_schedule, loss_fn, net):  # 保存网络训练的参数信息
    with open(os.path.join(logs_root, 'params_logs'), 'w') as f:
        f.write(f'batch_size={batch_size}\n')
        f.write(f'lr={lr}\n')
        f.write(f'wd={l2_alpha}\n')
        f.write(f'patience={patience}\n')
        f.write(f'is_pretrained={is_pretrained},weight_dir={weight_dir}\n')
        f.write(f'strat_epoch={strat_epoch},end_epoch={end_epoch}\n')
        f.write(f'train_trans={train_trans}\nval_trans={val_trans}\n')
        f.write(f'optim_fn={optim_fn}\nlr_schedule={lr_schedule}\n')
        f.write(f'loss_fn={loss_fn}\nnet={net}\n')


def get_val_labels(labels, text_labels):
    """将数值标签转换为文本标签"""
    return [text_labels[int(i)] for i in labels]


def predict_label(net, test_iter, num_rows, num_cols, text_labels, scale=1.5):
    """预测标签并显示图像，支持调整子图大小"""
    for X, y in test_iter:
        break  # 取出一个 batch 进行预测

    batch_size = num_rows * num_cols  # 计算需要显示的图片数量
    X, y = X[:batch_size], y[:batch_size]  # 只取前 batch_size 个样本

    trues = get_val_labels(y, text_labels)
    preds = get_val_labels(net(X).argmax(axis=1), text_labels)
    titles = [f"True: {true}\nPred: {pred}" for true, pred in zip(trues, preds)]

    show_images(X, num_rows, num_cols, titles=titles, scale=scale)


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """使用 Matplotlib 显示图像，支持调整窗口大小"""
    figsize = (num_cols * scale, num_rows * scale)  # 计算窗口大小
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)  # 创建子图
    axes = axes.flatten() if num_rows > 1 or num_cols > 1 else [axes]  # 处理只有一个子图的情况

    for i, (ax, img) in enumerate(zip(axes, imgs)):
        img = img.permute(1, 2, 0).numpy()  # 转换维度 (C, H, W) -> (H, W, C)
        ax.imshow(img, cmap='gray' if img.shape[-1] == 1 else None)  # 适应灰度图或RGB图
        ax.set_xticks([])
        ax.set_yticks([])
        if titles:
            ax.set_title(titles[i], fontsize=10)

    # 隐藏未使用的子图
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def get_val_labels_col(labels, text_labels):  # 将数值标签转换为文本标签
    return [text_labels[int(i)] for i in labels]


def predict_label_col(net, test_iter, num_rows, num_cols, text_labels):
    batch_size = num_rows * num_cols  # 计算需要的批量大小
    for X, y in test_iter:
        break  # 只取一个批次数据

    trues = get_val_labels_col(y[:batch_size], text_labels)
    preds = get_val_labels_col(net(X[:batch_size]).argmax(axis=1), text_labels)
    titles = [f"True: {true}\nPred: {pred}" for true, pred in zip(trues, preds)]

    show_images_col(X[:batch_size], num_rows, num_cols, titles=titles)


def show_images_col(imgs, num_rows, num_cols, titles=None, scale=2.0):
    figsize = (num_cols * scale, num_rows * scale)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()

    for i, (ax, img) in enumerate(zip(axes, imgs)):
        img = img.permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        if titles:
            ax.set_title(titles[i], fontsize=10)

    for j in range(i + 1, len(axes)):  # 处理多余的子图
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


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


def my_collate_fn(batch, transform):  # 将一个batch堆叠成一个批次进行transforms,使用lambda表达式调用
    images = [item[0] for item in batch]  # batch是一个列表，每个元素是一个元组(img, label)
    labels = [item[1] for item in batch]
    images = torch.stack([transform(img) for img in images])  # 对每个图片应用transform
    labels = torch.tensor(labels)
    return images, labels


class MyDatasetGeneratorTIFF(torch.utils.data.Dataset):  # 生成dataset的类,读取的是离散的数据集
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
        with Image.open(img_path) as img:  # 读取单通道（灰度）图像
            img = img.convert("L")  # 转换为灰度图像（单通道）
        if self.transform is not None:
            img = self.transform(img)
        return img, img_label

    def __len__(self):
        return len(self.imgs_dir)


def generate_npy_from_tiff(data_path, data_npy_path):  # 将数据转为Tensor导入内存,加速训练
    # !!!!!   不能在这里面进行预处理，否则与出事的数据集没有区别，还是会过拟合   !!!!!
    train_path = os.path.join(data_path, 'train')
    val_path = os.path.join(data_path, 'val')
    os.makedirs(data_npy_path, exist_ok=True)
    train_set = MyDatasetGeneratorTIFF(train_path, torchvision.transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), shuffle=False)
    for data in train_loader:
        train_features, train_targets = data
        train_features = train_features.numpy()
        train_targets = train_targets.numpy()
    os.makedirs(name=os.path.join(data_npy_path), exist_ok=True)
    np.save(os.path.join(data_npy_path, 'train_features'), train_features)
    np.save(os.path.join(data_npy_path, 'train_targets'), train_targets)
    print('> train_date.npy has finished generate')
    val_set = MyDatasetGeneratorTIFF(val_path, torchvision.transforms.ToTensor())
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=len(val_set), shuffle=False)
    for data in val_loader:
        val_features, val_targets = data
        val_features = val_features.numpy()
        val_targets = val_targets.numpy()
    np.save(os.path.join(data_npy_path, 'val_features'), val_features)
    np.save(os.path.join(data_npy_path, 'val_targets'), val_targets)
    print('> val_date.npy has finished generate')
