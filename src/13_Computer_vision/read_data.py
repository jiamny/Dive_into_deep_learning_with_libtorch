import os
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch import jit
import torch
import torchvision
from d2l import torch as d2l


def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    print(factor)
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
        print(center)
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))

    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)

    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))

    weight[range(in_channels), range(out_channels), :, :] = filt

    return weight


def read_voc_images(voc_dir, is_train=True):
    """Read all VOC feature and label images.

    Defined in :numref:`sec_semantic_segmentation`"""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'SegmentationClass', f'{fname}.png'), mode))
    return features, labels


VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']


def voc_colormap2label():
    """Build the mapping from RGB to class indices for VOC labels.

    Defined in :numref:`sec_semantic_segmentation`"""
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label


def voc_label_indices(colormap, colormap2label):
    """Map any RGB values in VOC labels to their class indices.

    Defined in :numref:`sec_semantic_segmentation`"""
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]


def voc_rand_crop(feature, label, height, width):
    """Randomly crop both feature and label images.

    Defined in :numref:`sec_semantic_segmentation`"""
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label


class VOCSegDataset(torch.utils.data.Dataset):
    """A customized dataset to load the VOC dataset.

    Defined in :numref:`sec_semantic_segmentation`"""

    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.float() / 255)

    def filter(self, imgs):
        return [img for img in imgs if (
                img.shape[1] >= self.crop_size[0] and
                img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)


def load_data_voc(batch_size, crop_size):
    """Load the VOC semantic segmentation dataset.

    Defined in :numref:`sec_semantic_segmentation`"""
    voc_dir = '/home/stree/git/D2L_pytorch_cpp/data/VOCdevkit/VOC2012'
    # d2l.download_extract('voc2012', os.path.join('VOCdevkit', 'VOC2012'))
    num_workers = d2l.get_dataloader_workers()
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        drop_last=True, num_workers=num_workers)
    return train_iter, test_iter


torch.cuda.is_available = lambda: False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

voc_dir = "/home/stree/git/Dive_into_deep_learning_with_libtorch/data/VOCdevkit/VOC2012"
is_train = True

# Download and load the pre-trained model
net = jit.load("pretrained_resnet18.pt")
net.to(device)

net = nn.Sequential(*list(net.children())[:-2])

X = torch.rand(size=(1, 3, 320, 480))
print(net(X).shape)

# Set upgrading the gradients to False
num_classes = 21
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                                    kernel_size=64, padding=16, stride=32))

W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W)

batch_size, crop_size = 32, (320, 480)
# train_iter, test_iter = d2l.load_data_voc(batch_size, crop_size)

print(d2l.get_dataloader_workers())
num_workers = 1
train_iter = torch.utils.data.DataLoader(
    VOCSegDataset(True, crop_size, voc_dir), batch_size,
    shuffle=True, drop_last=True, num_workers=num_workers)
'''
test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        drop_last=True, num_workers=num_workers)
'''


def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)


num_epochs, lr, wd = 5, 0.001, 1e-3
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)

# d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)

num_batches = len(train_iter)

# net = nn.DataParallel(net, device_ids=devices).to(devices[0])
# for epoch in range(num_epochs):

for i, (X, y) in enumerate(train_iter):
    print(i)
    if i == 0:
        print(x.shape)
        print(y)
        if isinstance(X, list):
            # Required for BERT fine-tuning (to be covered later)
            X = [x.to(devices[) for x in X]
            else:
            X = X.to(devices[0])
            y = y.to(devices[0])
            net.train()
            trainer.zero_grad()
            pred = net(X)
            l = loss(pred, y)
            l.sum().backward()
            trainer.step()
            train_loss_sum = l.sum()
            train_acc_sum = d2l.accuracy(pred, y)
        return train_loss_sum, train_acc_sum
    # l, acc = train_batch_ch13(
    #    net, features, labels, loss, trainer, devices)
