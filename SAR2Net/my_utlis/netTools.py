"""
常用的方法放在这里
"""
import torch
import os
from torch import nn
import random
import numpy as np
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


class Animator111:  # 动态绘制训练过程中的损失和准确率曲线
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

        if is_earlyStop:  # 早停
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                counter = 0
            else:
                counter += 1
            if counter >= patience:
                print("> Early stopping triggered.")
                break

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


def get_val_labels(labels, text_labels):  # 将数值标签转换为文本标签
    return [text_labels[int(i)] for i in labels]


def predict_label(net, test_iter, batch_size, text_labels):  # 预测标签并显示图像
    for X, y in test_iter:
        break
    trues = get_val_labels(y, text_labels)
    preds = get_val_labels(net(X).argmax(axis=1), text_labels)
    titles = [f"True: {true}\nPred: {pred}" for true, pred in zip(trues, preds)]
    show_images(X[0:batch_size], 1, batch_size, titles=titles[0:batch_size])


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  # 使用 matplotlib 显示图像
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        img = img.permute(1, 2, 0).numpy()  # 将通道维度从 (C, H, W) 转换为 (H, W, C)
        ax.imshow(img)  # CIFAR-10 是 RGB 图像，不需要 cmap='gray'
        ax.set_xticks([])  # 隐藏x轴刻度
        ax.set_yticks([])  # 隐藏y轴刻度
        if titles:
            ax.set_title(titles[i])
    plt.tight_layout()
    plt.show()
