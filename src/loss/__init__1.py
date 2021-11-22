import os
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cv2

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.ndimage import gaussian_filter

from numpy.lib.stride_tricks import as_strided as ast
class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckp):
        super(Loss, self).__init__()
        print('Preparing loss function:')

        self.n_GPUs = args.n_GPUs
        self.loss = []
        self.loss_module = nn.ModuleList()
        self.epoch_tmp=0
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type.find('VGG') >= 0:
                module = import_module('loss.vgg')
                loss_function = getattr(module, 'VGG')(
                    loss_type[3:],
                    rgb_range=args.rgb_range
                )
            elif loss_type.find('GAN') >= 0:
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(
                    args,
                    loss_type
                )

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )
            if loss_type.find('GAN') >= 0:
                self.loss.append({'type': 'DIS', 'weight': 1, 'function': None})

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.log = torch.Tensor()

        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)
        if args.precision == 'half': self.loss_module.half()
        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(args.n_GPUs)
            )

        if args.load != '': self.load(ckp.dir, cpu=args.cpu)

    def forward(self, sr, hr):
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()
            elif l['type'] == 'DIS':
                self.log[-1, i] += self.loss[i - 1]['function'].loss

        #修改了损失函数
        # loss_sum = self.lossFcn(sr, hr)
        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log[-1, -1] += loss_sum.item()

        return loss_sum

    def total_variation_loss(self, generated_images):
        return torch.add(
            torch.sum(torch.abs(generated_images[:, :, :, :-1] - generated_images[:, :, :, 1:])),
            torch.sum(torch.abs(generated_images[:, :, :-1, :] - generated_images[:, :, 1:, :]))
        )

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, n_batches):
        self.log[-1].div_(n_batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))

        return ''.join(log)

    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(apath, 'loss_{}.pdf'.format(l['type'])))
            plt.close(fig)

    def get_loss_module(self):
        if self.n_GPUs == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step()

    def lossFcn(self, pred_img, ground_truth, coeff_basic=1.0):

        self.coeff_basic = coeff_basic
        self.loss_basic = self.LossBasic(pred_img, ground_truth)
        return self.coeff_basic * self.loss_basic#+0.2*self.psnrloss

    def LossBasic(self, pred, ground_truth):
        basic_L1 = False
        factor_gradient = 1.0
        #print(pred.shape)
        #print(pred)
        #mean = np.mean(cv2.cvtColor(pred,cv2.COLOR_RGB2GRAY))
        #print(mean)
        #gamma_val = math.log10(0.5) / math.log10(mean / 255)  # 公式计算gamma


        self.l1_loss = nn.L1Loss()
        self.epoch_tmp += 1
        if not basic_L1:
            self.basic_loss = nn.MSELoss()
        elif self.epoch_tmp > 100:
            self.basic_loss = nn.L1Loss()
        #return self.basic_loss(pred, ground_truth) + factor_gradient * self.l1_loss(self.gamma_trans(pred, gamma_val), ground_truth)

        return self.basic_loss(pred, ground_truth) + factor_gradient * self.l1_loss(pred,
                                                                                ground_truth)
    def gamma_trans(self, img, gamma):  # gamma函数处理

        gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
        return cv2.LUT(img, gamma_table)  # 图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。

    def nothing(self, x):
        pass



    def TensorGradient(self, img, L1=True):
        w, h = img.size(-2), img.size(-1)
        l = F.pad(img, [1, 0, 0, 0])
        r = F.pad(img, [0, 1, 0, 0])
        u = F.pad(img, [0, 0, 1, 0])
        d = F.pad(img, [0, 0, 0, 1])
        if L1:
            return torch.abs((l - r)[..., 0:w, 0:h]) + torch.abs((u - d)[..., 0:w, 0:h])
        else:
            return torch.sqrt(
                torch.pow((l - r)[..., 0:w, 0:h], 2) + torch.pow((u - d)[..., 0:w, 0:h], 2)
            )
    def PsnrLoss(self,img1,img2):
        img1 = img1.cuda().data.cpu().numpy()
        img2 = img2.cuda().data.cpu().numpy()
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))



