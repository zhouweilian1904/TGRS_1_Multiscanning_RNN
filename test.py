import spectral
import numpy as np
import torch
import torch.utils
import torch.utils.data
import os
from tqdm import tqdm
from spectral import *
from scipy.io import loadmat
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
from utils import open_file
import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.nn import init
# utils
import math
import os
import datetime
import numpy as np
from sklearn.externals import joblib
from tqdm import tqdm
from utils import grouper, sliding_window, count_sliding_window,\
                  camel_to_snake
import matplotlib.pyplot as plt
import matplotlib.image as mping
from torchsummary import summary
from torchvision.transforms import ToPILImage

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import matplotlib.pyplot as plt

x=open_file('/home/kamata/文档/DeepHyperX/DeepHyperX-master/Datasets/PCA_PaviaU_3/PCA_PaviaU_3.mat')
print(x.keys())

#
# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.conv1 = nn.Conv2d(1, 3, 3, 1, 1)
#         self.pool1 = nn.MaxPool2d(2)
#         self.conv2 = nn.Conv2d(3, 6, 3, 1, 1)
#         self.pool2 = nn.MaxPool2d(2)
#
#         self.conv_trans1 = nn.ConvTranspose2d(6, 3, 4, 2, 1)
#         self.conv_trans2 = nn.ConvTranspose2d(3, 1, 4, 2, 1)
#
#     def forward(self, x):
#         x = F.relu(self.pool1(self.conv1(x)))
#         x = F.relu(self.pool2(self.conv2(x)))
#         x = F.relu(self.conv_trans1(x))
#         x = self.conv_trans2(x)
#         return x
#
#
# dataset = datasets.MNIST(
#     root='PATH',
#     transform=transforms.ToTensor()
# )
# loader = DataLoader(
#     dataset,
#     num_workers=2,
#     batch_size=8,
#     shuffle=True
# )
#
# model = MyModel()
# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
#
# epochs = 1
# for epoch in range(epochs):
#     for batch_idx, (data, target) in enumerate(loader):
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, data)
#         loss.backward()
#         optimizer.step()
#
#         print('Epoch {}, Batch idx {}, loss {}'.format(
#             epoch, batch_idx, loss.item()))
#
#
# def normalize_output(img):
#     img = img - img.min()
#     img = img / img.max()
#     return img
#
#
# # Plot some images
# idx = torch.randint(0, output.size(0), ())
# pred = normalize_output(output[idx, 0])
# img = data[idx, 0]
#
# fig, axarr = plt.subplots(1, 2)
# axarr[0].imshow(img.detach().numpy())
# axarr[1].imshow(pred.detach().numpy())
#
# # Visualize feature maps
# activation = {}
#
#
# def get_activation(name):
#     def hook(model, input, output):
#         activation[name] = output.detach()
#
#     return hook
#
#
# model.conv1.register_forward_hook(get_activation('conv1'))
# data, _ = dataset[0]
# data.unsqueeze_(0)
# output = model(data)
#
# act = activation['conv1'].squeeze()
# fig, axarr = plt.subplots(act.size(0))
# for idx in range(act.size(0)):
#     axarr[idx].imshow(act[idx])
x=np.array([[[[[0.0841, 0.1374, 0.1331, 0.1233, 0.0986],
           [0.1078, 0.1021, 0.0639, 0.0591, 0.0604],
           [0.1350, 0.0809, 0.0188, 0.0576, 0.1057],
           [0.1353, 0.1281, 0.1700, 0.1386, 0.0703],
           [0.2111, 0.1714, 0.1547, 0.0896, 0.1361]],

          [[0.2074, 0.2222, 0.2159, 0.2343, 0.2350],
           [0.1747, 0.2098, 0.1852, 0.1592, 0.2322],
           [0.2336, 0.2495, 0.2503, 0.1926, 0.2367],
           [0.2268, 0.2879, 0.3101, 0.1984, 0.2369],
           [0.2887, 0.3008, 0.3255, 0.2803, 0.3040]],

          [[2.5650, 2.3709, 2.2286, 2.0477, 2.2417],
           [2.3109, 2.1477, 2.3034, 2.3152, 2.3628],
           [2.4253, 2.1610, 2.2112, 1.8341, 1.8436],
           [2.3014, 2.0896, 2.1906, 2.0610, 1.9500],
           [1.9721, 1.7858, 2.1392, 2.1503, 2.0452]]]]])

print(x.squeeze().shape)
x = x.squeeze()

img1 = x[0,:,:]
img2 = x[1,:,:]
img3 = x[2,:,:]
plt.subplot(1,3,1)
plt.imshow(img1)
plt.subplot(1,3,2)
plt.imshow(img2)
plt.subplot(1,3,3)
plt.imshow(img3)
plt.show()
# from torchvision.transforms import ToPILImage
# y = ToPILImage(x)
# y.imshow()