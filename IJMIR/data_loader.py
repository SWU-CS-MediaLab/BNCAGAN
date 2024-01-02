#-*-coding:utf-8-*-

from pathlib import Path
from itertools import chain
import os

from munch import Munch
from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms


def listdir(dname):
    # 获取图片名称列表
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames


class DefaultDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = listdir(root)
        self.samples.sort()
        self.transform = transform
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]
        img = Image.open(fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)


class ReferenceDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = self._make_dataset(root)
        self.transform = transform

    def _make_dataset(self, root):
        # 返回指定的文件夹包含的文件或文件夹的名字的列表
        # ./data/fivek/train
        # ./data/fivek/val
        domains = os.listdir(root)
        fnames, fnames2 = [], []
        # train
        # idx:0, domain:exp
        # idx:1, domain:raw
        # val
        # idx:0, domain:label
        # idx:1, domain:raw
        for idx, domain in enumerate(sorted(domains)):
            # os.path.join()路径拼接
            # class_dir:./data/fivek/train/exp等共四种
            class_dir = os.path.join(root, domain)
            # listdir()用于返回指定的文件夹包含的文件或文件夹的名字的列表
            # cls_fnames是每个图片的路径和名称
            cls_fnames = listdir(class_dir)
            # fnames里是idx为0的路径，也就是exp和label的
            # fnames2里是idx为1的路径，也就是raw的
            if idx == 0:
                fnames += cls_fnames
            elif idx == 1:
                fnames2 += cls_fnames
        # zip()打包为元组列表
        # list() 方法用于将元组转换为列表
        return list(zip(fnames, fnames2))

    def __getitem__(self, index):
        fname, fname2 = self.samples[index]
        name = str(fname2)
        # img_name就是去掉了.jpg
        img_name, _ = name.split('.', 1)
        # img_name去掉了前面的路径，只剩名称
        _, img_name = img_name.rsplit('/', 1)
        # 转换成RGB后读出图像
        img = Image.open(fname).convert('RGB')
        img2 = Image.open(fname2).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)
        return img, img2, img_name

    def __len__(self):
        return len(self.samples)


def get_train_loader(root, img_size=512, resize_size=256, batch_size=1, shuffle=True, num_workers=0, drop_last=True):

    transform = transforms.Compose([
        # 从图片中随机裁剪出尺寸为img_size(512)的图片，如果有padding，那么先进行padding，再随机裁剪img_size大小的图片。
        transforms.RandomCrop(img_size),
        # 把图片缩放到(resize_size, resize_size)大小(256, 256)
        transforms.Resize([resize_size, resize_size]),
        # 一半的图片会被水平翻转
        transforms.RandomHorizontalFlip(p=0.5),
        # 一半的图片会被垂直翻转
        transforms.RandomVerticalFlip(p=0.5),
        # 将PIL Image或者 ndarray 转换为tensor，并且归一化至(0, 1)
        transforms.ToTensor(),
        # 把(0, 1)变换到(-1, 1)
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    dataset = ReferenceDataset(root, transform)

    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)


def get_test_loader(root, img_size=512, batch_size=1, shuffle=False, num_workers=0):
    transform = transforms.Compose([
        # transforms.CenterCrop(img_size),
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    dataset = ReferenceDataset(root, transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True)


class InputFetcher:
    def __init__(self, loader):
        self.loader = loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _fetch_refs(self):
        try:
            x, y, name = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y, name = next(self.iter)
        return x, y, name

    def __next__(self):
        x, y, img_name = self._fetch_refs()
        x, y = x.to(self.device), y.to(self.device)
        inputs = Munch(img_exp=x, img_raw=y, img_name=img_name)
        
        return inputs

