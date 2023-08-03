# -*- coding: utf-8 -*-
# Torch
import sys
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn.functional as F
import torch
torch.set_grad_enabled(True)
torch.autograd.set_detect_anomaly(True)
import seaborn as sns
import visdom
# from main import viz as viz1
vis = visdom.Visdom()
sns.set()
sns.set_style('whitegrid')
sns.set_context('notebook')
from torch.nn import init
import visdom
viz = visdom.Visdom(env='regression', port=8097)
viz_tsne = visdom.Visdom(env='tsne', port=8097)
import math
import os
import datetime
import numpy as np
import joblib
from torch.autograd import Variable
from tqdm import tqdm
from utils import grouper, sliding_window, count_sliding_window,\
                  camel_to_snake, calculate_psnr
import matplotlib.patches as patches
from sklearn.decomposition import PCA
from einops import rearrange, repeat, reduce
# from VitNew4_3d_learned_absolute_pos_encoding import patch_3D
# from VitNew4_3d_learned_absolute_pos_encoding import get_pos_encode2
from center_loss import CenterLoss
from TGRS_3_vit_3d import TGRS_3
from VitNew4_3d_learned_absolute_pos_encoding import multiscan
# import dhg
import optuna
import torch.optim as optim
from sub_band_partialized import subIP
from TGRS_2 import multiTrans
from TGRS_3_patialized import partialized
from SSFTTnet import SSFTTnet
from yang import Yangnew
from vit_pytorch.vit_3d import ViT3d
from utils import perimeter_scan_batch, diagonal_flatten_image
import skimage.metrics as metrics

def get_model(name, **kwargs):
    """
    Instantiate and obtain a model with adequate hyperparameters

    Args:
        name: string of the model name
        kwargs: hyperparameters
    Returns:
        model: PyTorch network
        optimizer: PyTorch optimizer
        criterion: PyTorch loss Function
        kwargs: hyperparameters with sane defaults
    """
    # device = kwargs.setdefault('device', torch.device('cuda:0'))
    device = kwargs.setdefault("device", torch.device("cuda"))
    n_classes = kwargs['n_classes']
    n_bands = kwargs['n_bands']
    weights = torch.ones(n_classes)
    weights[torch.LongTensor(kwargs['ignored_labels'])] = 0.
    weights = weights.to(device)
    weights = kwargs.setdefault('weights', weights)

    if name == 'nn':
        kwargs.setdefault('patch_size', 1)
        center_pixel = True
        model = Baseline(n_bands, n_classes,
                         kwargs.setdefault('dropout', False))
        lr = kwargs.setdefault('learning_rate', 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        criterion = (nn.CrossEntropyLoss(weight=kwargs['weights']).to(device, dtype=torch.float32),
                     lambda rec, data: F.mse_loss(rec, data[:, :].squeeze()))
        kwargs.setdefault('epoch', 100)
        kwargs.setdefault('batch_size', 100)

    elif name == 'hamida':
        patch_size = kwargs.setdefault('patch_size', 11)
        center_pixel = True
        model = HamidaEtAl(n_bands, n_classes, patch_size=patch_size)
        # lr = kwargs.setdefault('learning_rate', 0.01)
        optimizer = optim.Adam(model.parameters())
        kwargs.setdefault('batch_size', 100)
        # criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        criterion = (nn.CrossEntropyLoss(weight=kwargs['weights']).to(device, dtype=torch.float32),
                     lambda rec, data: F.mse_loss(rec, data[:, :, :, patch_size // 2, patch_size // 2].squeeze()))
    elif name == 'lee':
        kwargs.setdefault('epoch', 200)
        patch_size = kwargs.setdefault('patch_size', 24)
        center_pixel = False
        model = LeeEtAl(n_bands, n_classes)
        lr = kwargs.setdefault('learning_rate', 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        criterion = (nn.CrossEntropyLoss(weight=kwargs['weights']).to(device, dtype=torch.float32),
                     lambda rec, data: F.mse_loss(rec, data[:, :, :, patch_size // 2, patch_size // 2].squeeze()))
    elif name == 'chen':
        patch_size = kwargs.setdefault('patch_size', 27)
        center_pixel = True
        model = ChenEtAl(n_bands, n_classes, patch_size=patch_size)
        # lr = kwargs.setdefault('learning_rate', 0.003)
        optimizer = optim.Adam(model.parameters())
        # criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        kwargs.setdefault('epoch', 200)
        kwargs.setdefault('batch_size', 100)
        criterion = (nn.CrossEntropyLoss(weight=kwargs['weights']).to(device, dtype=torch.float32),
                     lambda rec, data: F.mse_loss(rec, data[:, :, :, patch_size // 2, patch_size // 2].squeeze()))
    elif name == 'li':
        patch_size = kwargs.setdefault('patch_size', 11)
        center_pixel = True
        model = LiEtAl(n_bands, n_classes, n_planes=16, patch_size=patch_size)
        lr = kwargs.setdefault('learning_rate', 0.01)
        optimizer = optim.Adam(model.parameters())
        epoch = kwargs.setdefault('epoch', 200)
        # criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        criterion = (nn.CrossEntropyLoss(weight=kwargs['weights']).to(device, dtype=torch.float32),
                     lambda rec, data: F.mse_loss(rec, data[:, :, :, patch_size // 2, patch_size // 2].squeeze()))
        # kwargs.setdefault('scheduler', optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epoch // 2, (5 * epoch) // 6], gamma=0.1))
    elif name == "hu":
        kwargs.setdefault("patch_size", 1)
        center_pixel = True
        model = HuEtAl(n_bands, n_classes)
        # From what I infer from the paper (Eq.7 and Algorithm 1), it is standard SGD with lr = 0.01
        lr = kwargs.setdefault("learning_rate", 0.01)
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "he":
        # We train our model by AdaGrad [18] algorithm, in which
        # the base learning rate is 0.01. In addition, we set the batch
        # as 40, weight decay as 0.01 for all the layers
        # The input of our network is the HSI 3D patch in the size of 7×7×Band
        patch_size = kwargs.setdefault("patch_size", 11)
        kwargs.setdefault("batch_size", 40)
        lr = kwargs.setdefault("learning_rate", 0.01)
        center_pixel = True
        model = HeEtAl(n_bands, n_classes, patch_size=kwargs["patch_size"])
        # For Adagrad, we need to load the model on GPU before creating the optimizer
        model = model.to(device)
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=0.01)
        # criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        criterion = (nn.CrossEntropyLoss(weight=kwargs['weights']).to(device, dtype=torch.float32),
                     lambda rec, data: F.mse_loss(rec, data[:, :, :, patch_size // 2, patch_size // 2].squeeze()))
    elif name == 'luo':
        # All  the  experiments  are  settled  by  the  learning  rate  of  0.1,
        # the  decay  term  of  0.09  and  batch  size  of  100.
        patch_size = kwargs.setdefault('patch_size', 3)
        kwargs.setdefault('batch_size', 100)
        lr = kwargs.setdefault('learning_rate', 0.1)
        center_pixel = True
        model = LuoEtAl(n_bands, n_classes, patch_size=kwargs['patch_size'])
        optimizer = optim.Adam(model.parameters())
        # criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        criterion = (nn.CrossEntropyLoss(weight=kwargs['weights']).to(device, dtype=torch.float32),
                     lambda rec, data: F.mse_loss(rec, data[:, :, :, patch_size // 2, patch_size // 2].squeeze()))
    elif name == 'sharma':
        # We train our S-CNN from scratch using stochastic gradient descent with
        # momentum set to 0.9, weight decay of 0.0005, and with a batch size
        # of 60.  We initialize an equal learning rate for all trainable layers
        # to 0.05, which is manually decreased by a factor of 10 when the validation
        # error stopped decreasing. Prior to the termination the learning rate was
        # reduced two times at 15th and 25th epoch. [...]
        # We trained the network for 30 epochs
        kwargs.setdefault('batch_size', 60)
        epoch = kwargs.setdefault('epoch', 30)
        lr = kwargs.setdefault('lr', 0.05)
        center_pixel = True
        # We assume patch_size = 64
        kwargs.setdefault('patch_size', 64)
        model = SharmaEtAl(n_bands, n_classes, patch_size=kwargs['patch_size'])
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        criterion = (nn.CrossEntropyLoss(weight=kwargs['weights']).to(device, dtype=torch.float32),
                     lambda rec, data: F.mse_loss(rec, data[:, :, :, patch_size // 2, patch_size // 2].squeeze()))
        # kwargs.setdefault('scheduler', optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epoch // 2, (5 * epoch) // 6], gamma=0.1))
    elif name == "liu":
        kwargs["supervision"] = "semi"
        # "The learning rate is set to 0.001 empirically. The number of epochs is set to be 40."
        kwargs.setdefault("epoch", 40)
        lr = kwargs.setdefault("lr", 0.001)
        center_pixel = True
        patch_size = kwargs.setdefault("patch_size", 9)
        model = LiuEtAl(n_bands, n_classes, patch_size)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        # "The unsupervised cost is the squared error of the difference"
        criterion = (
            nn.CrossEntropyLoss(weight=kwargs["weights"]),
            lambda rec, data: F.mse_loss(
                rec, data[:, :, :, patch_size // 2, patch_size // 2].squeeze()
            ),
        )
    elif name == 'boulch':
        kwargs['supervision'] = 'semi'
        kwargs.setdefault('patch_size', 1)
        kwargs.setdefault('epoch', 100)
        lr = kwargs.setdefault('lr', 0.001)
        center_pixel = True
        model = BoulchEtAl(n_bands, n_classes)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = (nn.CrossEntropyLoss(weight=kwargs['weights']), lambda rec, data: F.mse_loss(rec, data.squeeze()))
    elif name =='zhoupatchdependency':
        kwargs.setdefault('patch_size',9)
        center_pixel = True
        kwargs.setdefault('epoch', 200)
        kwargs.setdefault('batch_size',100)
        lr = kwargs.setdefault('lr', 1.0)
        model = zhoupatchdependency(n_bands,n_classes)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    elif name == "mou":
        kwargs.setdefault("patch_size", 1)
        center_pixel = True
        kwargs.setdefault("epoch", 100)
        # "The RNN was trained with the Adadelta algorithm [...] We made use of a
        # fairly  high  learning  rate  of  1.0  instead  of  the  relatively  low
        # default of  0.002 to  train the  network"
        lr = kwargs.setdefault("lr", 1.0)
        model = MouEtAl(n_bands, n_classes)
        # For Adadelta, we need to load the model on GPU before creating the optimizer
        model = model.to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
    elif name == 'zhouSpectralAttention':
        kwargs['supervision'] = 'semi'
        kwargs.setdefault('patch_size', 1)
        center_pixel = True
        kwargs.setdefault('epoch', 200)
        kwargs.setdefault('batch_size', 100)
        # lr = kwargs.setdefault('lr', 1.0)
        model = zhouSpectralAttention(n_bands, n_classes)
        # For Adadelta, we need to load the model on GPU before creating the optimizer
        model = model.to(device)
        optimizer = optim.Adam(model.parameters())
        criterion = (nn.CrossEntropyLoss(weight=kwargs['weights']), lambda rec, data: F.mse_loss(rec, data.squeeze()))
    elif name =='zhouConstraintRNN':
        kwargs.setdefault('patch_size',5)
        center_pixel = True
        kwargs.setdefault('epoch', 200)
        kwargs.setdefault('batch_size',100)
        lr = kwargs.setdefault('lr', 1.0)
        model = zhouConstraintRNN(n_bands,n_classes)
        model = model.to(device)
        optimizer = optim.Adadelta(model.parameters(),lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    elif name == 'zhouOneDRNN':
        patch_size = kwargs.setdefault('patch_size', 5)
        center_pixel = True
        kwargs.setdefault('epoch', 200)
        kwargs.setdefault('batch_size', 100)
        lr = kwargs.setdefault('lr', 1.0)
        model = zhouOneDRNN(n_bands, n_classes,patch_size=patch_size)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters())
        # criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        criterion = (nn.CrossEntropyLoss(weight=kwargs['weights']).to(device, dtype=torch.float32),
                     lambda rec, data: F.mse_loss(rec, data[:, :, :, patch_size // 2, patch_size // 2].squeeze()))
    elif name == 'zhouFourDRNN':
        patch_size =kwargs.setdefault('patch_size', 5)
        center_pixel = True
        kwargs.setdefault('epoch', 200)
        kwargs.setdefault('batch_size', 100)
        lr = kwargs.setdefault('lr', 1.0)
        model = zhouFourDRNN(n_bands, n_classes,patch_size=patch_size)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters())
        # criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        criterion = (nn.CrossEntropyLoss(weight=kwargs['weights']).to(device, dtype=torch.float32),
                     lambda rec, data: F.mse_loss(rec, data[:, :, :, patch_size // 2, patch_size // 2].squeeze()))
    elif name == 'zhouEightDRNN':
        kwargs['supervision'] = 'full'
        patch_size = kwargs.setdefault('patch_size', 5)
        center_pixel = True
        kwargs.setdefault('epoch', 200)
        kwargs.setdefault('batch_size', 100)
        lr = kwargs.setdefault('lr', 1.0)
        model = zhouEightDRNN(n_bands, n_classes,patch_size=patch_size)
        model = model.to(device)
        optimizer = optim.AdamW(model.parameters())
        # criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        criterion = (nn.CrossEntropyLoss(weight=kwargs['weights']).to(device, dtype=torch.float32),
                     lambda rec, data: F.mse_loss(rec, data[:, :, :, patch_size // 2, patch_size // 2].squeeze()))
    elif name == 'multiTrans':
        kwargs['supervision'] = 'full'
        patch_size = kwargs.setdefault('patch_size', 7)
        center_pixel = True
        epoch = kwargs.setdefault('epoch', 200)
        batch_size = kwargs.setdefault('batch_size', 100)
        lr = kwargs.setdefault('lr', 0.001)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # center_loss = CenterLoss(num_classes=n_classes,feat_dim=n_bands,use_gpu=True ).to(device)
        model = multiTrans(n_bands, n_classes, patch_size=patch_size)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-3)
        # criterion = nn.CrossEntropyLoss(weight=kwargs['weights']).to(device, dtype=torch.float32)
        criterion = (nn.CrossEntropyLoss(weight=kwargs['weights']).to(device, dtype=torch.float32),
                     lambda rec, data: F.mse_loss(rec, data[:, :, :, patch_size // 2, patch_size // 2].squeeze()))
        # criterion = (nn.CrossEntropyLoss(weight=kwargs['weights']),
        #              lambda rec, target: center_loss(rec, target))
        # kwargs.setdefault('scheduler',
        #                   optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epoch // 2, (5 * epoch) // 6],
        #                                                  gamma=0.1))
    elif name == 'partialized':
        kwargs['supervision'] = 'full'
        patch_size = kwargs.setdefault('patch_size', 7)
        center_pixel = True
        epoch = kwargs.setdefault('epoch', 200)
        batch_size = kwargs.setdefault('batch_size', 100)
        lr = kwargs.setdefault('lr', 0.001)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = partialized(n_bands, n_classes, patch_size=patch_size)
        optimizer = optim.AdamW(model.parameters(), lr=lr, )
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        # criterion = (nn.CrossEntropyLoss(weight=kwargs['weights']),
        #              lambda rec, data: F.mse_loss(rec, data[:, :, :, patch_size // 2, patch_size // 2].squeeze()))
        kwargs.setdefault('scheduler',
                          optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epoch // 2, (5 * epoch) // 6],
                                                         gamma=0.1))
    elif name == 'SSFTT':
        kwargs['supervision'] = 'full'
        patch_size = kwargs.setdefault('patch_size', 7)
        center_pixel = True
        epoch = kwargs.setdefault('epoch', 200)
        batch_size = kwargs.setdefault('batch_size', 100)
        lr = kwargs.setdefault('lr', 0.001)
        model = SSFTTnet(n_bands, n_classes, patch_size=patch_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        optimizer = optim.AdamW(model.parameters(), lr=lr, )
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        # criterion = (nn.CrossEntropyLoss(weight=kwargs['weights']),
        #              lambda rec, data: F.mse_loss(rec, data[:, :, :, patch_size // 2, patch_size // 2].squeeze()))
        kwargs.setdefault('scheduler',
                          optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epoch // 2, (5 * epoch) // 6],
                                                         gamma=0.1))
    elif name == 'TGRS_3':
        kwargs['supervision'] = 'full'
        patch_size = kwargs.setdefault('patch_size', 7)
        center_pixel = True
        epoch = kwargs.setdefault('epoch', 200)
        batch_size = kwargs.setdefault('batch_size', 100)
        lr = kwargs.setdefault('lr', 0.001)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TGRS_3(n_bands, n_classes, batch_size = batch_size, patch_size=patch_size)
        # model = model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights']).to(device, dtype=torch.float32)
        # criterion = (nn.CrossEntropyLoss(weight=kwargs['weights']),
        #              lambda rec, data: F.mse_loss(rec, data[:, :, :, patch_size // 2, patch_size // 2].squeeze()))
        kwargs.setdefault('scheduler',
                          optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epoch // 2, (5 * epoch) // 6],
                                                         gamma=0.1))
    elif name == 'ViT3d':
        kwargs['supervision'] = 'full'
        patch_size = kwargs.setdefault('patch_size', 9)
        center_pixel = True
        epoch = kwargs.setdefault('epoch', 200)
        batch_size = kwargs.setdefault('batch_size', 64)
        lr = kwargs.setdefault('lr', 0.001)
        model = ViT3d(n_bands, n_classes, patch_size, batch_size)
        model = model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.9))
        criterion = (nn.CrossEntropyLoss(weight=kwargs['weights']).to(device, dtype=torch.float32),
                     lambda rec, data: F.mse_loss(rec, data[:, :, :, patch_size // 2, patch_size // 2].squeeze()))
        kwargs.setdefault('scheduler', torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5, verbose=True))
    elif name == 'yang':
        kwargs['supervision'] = 'full'
        patch_size = kwargs.setdefault('patch_size', 15)
        center_pixel = True
        lr = kwargs.setdefault('lr', 0.001)
        model = Yangnew(n_bands, n_classes, patch_size=15)
        # model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights']).to(device, dtype=torch.float)
        batch_size = kwargs.setdefault("batch_size", 100)
        epoch = kwargs.setdefault('epoch', 200)
        kwargs.setdefault('scheduler',
                          optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epoch // 2, (5 * epoch) // 6],
                                                         gamma=0.1))
    elif name == 'zhouEightDRNN_kamata_LSTM':
        patch_size =kwargs.setdefault('patch_size', 5)
        center_pixel = True
        kwargs.setdefault('epoch', 200)
        kwargs.setdefault('batch_size', 100)
        lr = kwargs.setdefault('lr', 1.0)
        model = zhouEightDRNN_kamata_LSTM(n_bands, n_classes,patch_size=patch_size)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    elif name == 'zhouEightDRNN_kamata_Transformer':
        patch_size = kwargs.setdefault('patch_size', 5)
        center_pixel = True
        kwargs.setdefault('epoch', 200)
        kwargs.setdefault('batch_size', 100)
        lr = kwargs.setdefault('lr', 1.0)
        model = zhouEightDRNN_kamata_Transformer(n_bands, n_classes,patch_size=patch_size)
        model = model.to(device)
        optimizer = optim.AdamW(model.parameters())
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    elif name == 'zhou_slidingLSTM_Trans':
        kwargs.setdefault('patch_size', 5)
        center_pixel = True
        kwargs.setdefault('epoch', 200)
        kwargs.setdefault('batch_size', 100)
        lr = kwargs.setdefault('lr', 1.0)
        model = zhou_slidingLSTM_Trans(n_bands, n_classes)
        model = model.to(device)
        optimizer = optim.AdamW(model.parameters())
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    elif name == 'zhou_dual_LSTM_Trans':
        patch_size = kwargs.setdefault('patch_size', 5)
        center_pixel = True
        epoch = kwargs.setdefault('epoch', 200)
        kwargs.setdefault('batch_size', 100)
        lr = kwargs.setdefault('lr', 1.0)
        model = zhou_dual_LSTM_Trans(n_bands, n_classes)
        model = model.to(device)
        optimizer = optim.AdamW(model.parameters())
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        # kwargs.setdefault('scheduler',
        #                   optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epoch // 2, (5 * epoch) // 6],
        #                                                  gamma=0.1))
    elif name == 'zhou_single_multi_scanning_Trans':
        patch_size = kwargs.setdefault('patch_size', 5)
        center_pixel = True
        kwargs.setdefault('epoch', 200)
        kwargs.setdefault('batch_size', 100)
        lr = kwargs.setdefault('lr', 1.0)
        model = zhou_single_multi_scanning_Trans(n_bands, n_classes)
        model = model.to(device)
        optimizer = optim.AdamW(model.parameters())
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    elif name == 'zhou_ICPR2022':
        patch_size = kwargs.setdefault('patch_size', 9)
        center_pixel = True
        kwargs.setdefault('epoch', 200)
        kwargs.setdefault('batch_size', 100)
        lr = kwargs.setdefault('lr', 1.0)
        model = zhouICPR2022(n_bands, n_classes)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    elif name == 'zhoumultiscanning':
        patch_size = kwargs.setdefault('patch_size', 5)
        center_pixel = True
        kwargs.setdefault('epoch', 200)
        kwargs.setdefault('batch_size', 100)
        lr = kwargs.setdefault('lr', 1.0)
        model = zhoumultiscanning(n_bands, n_classes)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    elif name == 'zhoumultiscanning_Trans':
        patch_size = kwargs.setdefault('patch_size', 5)
        center_pixel = True
        kwargs.setdefault('epoch', 200)
        kwargs.setdefault('batch_size', 100)
        lr = kwargs.setdefault('lr', 1.0)
        model = zhoumultiscanning_Trans(n_bands, n_classes)
        model = model.to(device)
        optimizer = optim.AdamW(model.parameters())
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    elif name == 'zhouSSViT':
        patch_size = kwargs.setdefault('patch_size', 9)
        center_pixel = True
        kwargs.setdefault('epoch', 200)
        kwargs.setdefault('batch_size', 100)
        lr = kwargs.setdefault('lr', 1.0)
        model = zhouSSViT(n_bands, n_classes)
        model = model.to(device)
        optimizer = optim.AdamW(model.parameters())
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    elif name == 'zhou3dvit':
        patch_size = kwargs.setdefault('patch_size', 9)
        center_pixel = True
        kwargs.setdefault('epoch', 200)
        kwargs.setdefault('batch_size', 100)
        lr = kwargs.setdefault('lr', 1.0)
        model = zhou3dvit(n_bands, n_classes)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    elif name == 'zhouICIP2022':
        patch_size = kwargs.setdefault('patch_size', 9)
        center_pixel = True
        kwargs.setdefault('epoch', 200)
        kwargs.setdefault('batch_size', 100)
        lr = kwargs.setdefault('lr', 1.0)
        model = zhouICIP2022(n_bands, n_classes)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    elif name == 'zhouEightDRNN_kamata_singleD':
        kwargs.setdefault('patch_size', 5)
        center_pixel = True
        kwargs.setdefault('epoch', 200)
        kwargs.setdefault('batch_size', 16)
        lr = kwargs.setdefault('lr', 1.0)
        model = zhouEightDRNN_kamata_singleD(n_bands, n_classes)
        model = model.to(device)
        optimizer = optim.AdamW(model.parameters())
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    elif name == 'zhou16dRNN':
        kwargs.setdefault('patch_size', 3)
        patch_size = kwargs.setdefault('patch_size', 3)
        center_pixel = True
        kwargs.setdefault('epoch', 200)
        kwargs.setdefault('batch_size', 100)
        # lr = kwargs.setdefault('lr', 1.0)
        model = zhou16dRNN(n_bands, n_classes)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    elif name == 'zhouRNNC':
        kwargs.setdefault('patch_size', 5)
        center_pixel = True
        kwargs.setdefault('epoch', 100)
        kwargs.setdefault('batch_size', 16)
        lr = kwargs.setdefault('lr', 0.1)
        model = zhouRNNC(n_bands, n_classes)
        model = model.to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    elif name == 'zhouDR':
        patch_size = kwargs.setdefault('patch_size', 9)
        center_pixel = True
        model = zhouDREtAl(n_bands, n_classes, n_planes=16, patch_size=patch_size)
        model = model.to(device)
        lr = kwargs.setdefault('learning_rate', 0.01)
        optimizer = optim.Adam(model.parameters())
        epoch = kwargs.setdefault('epoch', 200)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    elif name == 'subIP_semi':
        patch_size = kwargs.setdefault('patch_size', 9)
        kwargs['supervision'] = 'semi'
        # "The learning rate is set to 0.001 empirically. The number of epochs is set to be 40."
        center_pixel = True
        kwargs.setdefault('batch_size', 100)
        epoch = kwargs.setdefault('epoch', 200)
        lr = kwargs.setdefault('lr', 0.001)
        model = subIP(n_bands, n_classes, patch_size=patch_size)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters())
        # "The unsupervised cost is the squared error of the difference"
        # criterion = (nn.CrossEntropyLoss(weight=kwargs['weights']), lambda rec, data: F.mse_loss(rec, data[:,:,:,patch_size//2, patch_size//2].squeeze()))
        # criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        criterion = (nn.CrossEntropyLoss(weight=kwargs['weights']),
                     lambda rec, data: F.mse_loss(rec, data[:, :, :, patch_size // 2, patch_size // 2].squeeze()))
        kwargs.setdefault('scheduler',
                          optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epoch // 2, (5 * epoch) // 6],
                                                         gamma=0.1))


    else:
        raise KeyError("{} model is unknown.".format(name))

    model = model.to(device)
    epoch = kwargs.setdefault('epoch', 200)
    # kwargs.setdefault('scheduler', optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=epoch//4, verbose=True))
    # kwargs.setdefault('scheduler',
    #                   optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epoch // 2, (5 * epoch) // 6],
    #                                                  gamma=0.1))
    kwargs.setdefault('scheduler', None)
    kwargs.setdefault('batch_size', 100)
    kwargs.setdefault('supervision', 'full')
    kwargs.setdefault('flip_augmentation', False)
    kwargs.setdefault('radiation_augmentation', False)
    kwargs.setdefault('mixture_augmentation', False)
    kwargs['center_pixel'] = center_pixel
    return model, optimizer, criterion, kwargs

class Baseline(nn.Module):
    """
    Baseline network
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, dropout=False):
        super(Baseline, self).__init__()
        self.use_dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(input_channels, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, 2048)
        self.fc4 = nn.Linear(2048, n_classes)
        self.reg = nn.Linear(2048, input_channels)
        self.apply(self.weight_init)

        self.aux_loss_weight = 1

    def forward(self, x):
        print('x',x.shape)
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc3(x))
        if self.use_dropout:
            x = self.dropout(x)
        x_class = self.fc4(x)
        x_reg = self.reg(x)
        return x_class, x_reg

class HuEtAl(nn.Module):
    """
    Deep Convolutional Neural Networks for Hyperspectral Image Classification
    Wei Hu, Yangyu Huang, Li Wei, Fan Zhang and Hengchao Li
    Journal of Sensors, Volume 2015 (2015)
    https://www.hindawi.com/journals/js/2015/258619/
    """

    @staticmethod
    def weight_init(m):
        # [All the trainable parameters in our CNN should be initialized to
        # be a random value between −0.05 and 0.05.]
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            init.uniform_(m.weight, -0.05, 0.05)
            init.zeros_(m.bias)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_channels)
            x = self.pool(self.conv(x))
        return x.numel()

    def __init__(self, input_channels, n_classes, kernel_size=None, pool_size=None):
        super(HuEtAl, self).__init__()
        if kernel_size is None:
            # [In our experiments, k1 is better to be [ceil](n1/9)]
            kernel_size = math.ceil(input_channels / 9)
        if pool_size is None:
            # The authors recommand that k2's value is chosen so that the pooled features have 30~40 values
            # ceil(kernel_size/5) gives the same values as in the paper so let's assume it's okay
            pool_size = math.ceil(kernel_size / 5)
        self.input_channels = input_channels

        # [The first hidden convolution layer C1 filters the n1 x 1 input data with 20 kernels of size k1 x 1]
        self.conv = nn.Conv1d(1, 20, kernel_size)
        self.pool = nn.MaxPool1d(pool_size)
        self.features_size = self._get_final_flattened_size()
        # [n4 is set to be 100]
        self.fc1 = nn.Linear(self.features_size, 100)
        self.fc2 = nn.Linear(100, n_classes)
        self.apply(self.weight_init)
        self.aux_loss_weight = 1

    def forward(self, x):
        # [In our design architecture, we choose the hyperbolic tangent function tanh(u)]
        x = x.squeeze(dim=-1).squeeze(dim=-1)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = torch.tanh(self.pool(x))
        x = x.view(-1, self.features_size)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

class HamidaEtAl(nn.Module):
    """
    3-D Deep Learning Approach for Remote Sensing Image Classification
    Amina Ben Hamida, Alexandre Benoit, Patrick Lambert, Chokri Ben Amar
    IEEE TGRS, 2018
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8344565
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=11
                 , dilation=1):
        super(HamidaEtAl, self).__init__()
        # The first layer is a (3,3,3) kernel sized Conv characterized
        # by a stride equal to 1 and number of neurons equal to 20
        self.patch_size = patch_size
        self.input_channels = input_channels
        dilation = (dilation, 1, 1)

        if patch_size == 3:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=1)
        else:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=0)
        # Next pooling is applied using a layer identical to the previous one
        # with the difference of a 1D kernel size (1,1,3) and a larger stride
        # equal to 2 in order to reduce the spectral dimension
        self.pool1 = nn.Conv3d(
            20, 20, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))
        # Then, a duplicate of the first and second layers is created with
        # 35 hidden neurons per layer.
        self.conv2 = nn.Conv3d(
            20, 35, (3, 3, 3), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0))
        self.pool2 = nn.Conv3d(
            35, 35, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))
        # Finally, the 1D spatial dimension is progressively reduced
        # thanks to the use of two Conv layers, 35 neurons each,
        # with respective kernel sizes of (1,1,3) and (1,1,2) and strides
        # respectively equal to (1,1,1) and (1,1,2)
        self.conv3 = nn.Conv3d(
            35, 35, (3, 1, 1), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0))
        self.conv4 = nn.Conv3d(
            35, 35, (2, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))

        #self.dropout = nn.Dropout(p=0.5)

        self.features_size = self._get_final_flattened_size()
        # The architecture ends with a fully connected layer where the number
        # of neurons is equal to the number of input classes.
        self.fc = nn.Linear(self.features_size, n_classes)
        self.reg = nn.Linear(self.features_size, input_channels)
        self.apply(self.weight_init)
        self.aux_loss_weight = 1

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.features_size)
        #x = self.dropout(x)
        x_class = self.fc(x)
        x_reg = self.reg(x)
        return x_class, x_reg

class LeeEtAl(nn.Module):
    """
    CONTEXTUAL DEEP CNN BASED HYPERSPECTRAL CLASSIFICATION
    Hyungtae Lee and Heesung Kwon
    IGARSS 2016
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_uniform_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, in_channels, n_classes):
        super(LeeEtAl, self).__init__()
        # The first convolutional layer applied to the input hyperspectral
        # image uses an inception module that locally convolves the input
        # image with two convolutional filters with different sizes
        # (1x1xB and 3x3xB where B is the number of spectral bands)
        self.conv_3x3 = nn.Conv3d(
            1, 128, (in_channels, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv_1x1 = nn.Conv3d(
            1, 128, (in_channels, 1, 1), stride=(1, 1, 1), padding=0)
        self.conv_5_5 = nn.Conv3d(
            1, 128, (in_channels, 5, 5), stride=(1, 1, 1), padding=(0, 2, 2)
        )
        self.conv_7_7 = nn.Conv3d(
            1, 128, (in_channels, 7, 7), stride=(1, 1, 1), padding=(0, 3, 3)
        )

        # We use two modules from the residual learning approach
        # Residual block 1
        self.conv1 = nn.Conv2d(128+128+128+128, 128, (1, 1))
        self.conv2 = nn.Conv2d(128, 128, (1, 1))
        self.conv3 = nn.Conv2d(128, 128, (1, 1))

        # Residual block 2
        self.conv4 = nn.Conv2d(128, 128, (1, 1))
        self.conv5 = nn.Conv2d(128, 128, (1, 1))

        # The layer combination in the last three convolutional layers
        # is the same as the fully connected layers of Alexnet
        self.conv6 = nn.Conv2d(128, 128, (1, 1))
        self.conv7 = nn.Conv2d(128, 128, (1, 1))
        self.conv8 = nn.Conv2d(128, n_classes, (1, 1))

        self.lrn1 = nn.LocalResponseNorm(256)
        self.lrn2 = nn.LocalResponseNorm(128)

        # The 7 th and 8 th convolutional layers have dropout in training
        self.dropout = nn.Dropout(p=0.5)

        self.apply(self.weight_init)
        self.aux_loss_weight = 1

    def forward(self, x):
        print('1',x.shape)
        # Inception module
        x_3x3 = self.conv_3x3(x)
        #print('2',x_3x3.shape)
        x_1x1 = self.conv_1x1(x)
        x_5_5 = self.conv_5_5(x)
        x_7_7 = self.conv_7_7(x)
        #print('3',x_1x1.shape)
        x = torch.cat([x_7_7, x_5_5, x_3x3, x_1x1], dim=1)
        print('4', x.shape)
        # Remove the third dimension of the tensor
        x = torch.squeeze(x)
        print('5',x.shape)

        # Local Response Normalization
        x = F.relu(self.lrn1(x))

        # First convolution
        x = self.conv1(x)

        # Local Response Normalization
        x = F.relu(self.lrn2(x))

        # First residual block
        x_res = F.relu(self.conv2(x))
        x_res = self.conv3(x_res)
        x = F.relu(x + x_res)

        # Second residual block
        x_res = F.relu(self.conv4(x))
        x_res = self.conv5(x_res)
        x = F.relu(x + x_res)

        x = F.relu(self.conv6(x))
        x = self.dropout(x)
        x = F.relu(self.conv7(x))
        print(x.shape)
        x = self.dropout(x)
        x = self.conv8(x)
        print(x.shape)
        return x

class ChenEtAl(nn.Module):
    """
    DEEP FEATURE EXTRACTION AND CLASSIFICATION OF HYPERSPECTRAL IMAGES BASED ON
                        CONVOLUTIONAL NEURAL NETWORKS
    Yushi Chen, Hanlu Jiang, Chunyang Li, Xiuping Jia and Pedram Ghamisi
    IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2017
    """
    @staticmethod
    def weight_init(m):
        # In the beginning, the weights are randomly initialized
        # with standard deviation 0.001
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.normal_(m.weight, std=0.001)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=27, n_planes=32):
        super(ChenEtAl, self).__init__()
        self.input_channels = input_channels
        self.n_planes = n_planes
        self.patch_size = patch_size

        self.conv1 = nn.Conv3d(1, n_planes, (32, 4, 4))
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.conv2 = nn.Conv3d(n_planes, n_planes, (32, 4, 4))
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.conv3 = nn.Conv3d(n_planes, n_planes, (32, 4, 4))

        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.apply(self.weight_init)

        self.aux_loss_weight = 1
    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = x.view(-1, self.features_size)
        x = self.fc(x)
        return x

class LiEtAl(nn.Module):
    """
    SPECTRAL–SPATIAL CLASSIFICATION OF HYPERSPECTRAL IMAGERY
            WITH 3D CONVOLUTIONAL NEURAL NETWORK
    Ying Li, Haokui Zhang and Qiang Shen
    MDPI Remote Sensing, 2017
    http://www.mdpi.com/2072-4292/9/1/67
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.xavier_uniform_(m.weight.data)
            init.constant_(m.bias.data, 0)

    def __init__(self, input_channels, n_classes, n_planes=2, patch_size=11):
        super(LiEtAl, self).__init__()
        self.input_channels = input_channels
        self.n_planes = n_planes
        self.patch_size = patch_size

        # The proposed 3D-CNN model has two 3D convolution layers (C1 and C2)
        # and a fully-connected layer (F1)
        # we fix the spatial size of the 3D convolution kernels to 3 × 3
        # while only slightly varying the spectral depth of the kernels
        # for the Pavia University and Indian Pines scenes, those in C1 and C2
        # were set to seven and three, respectively
        self.conv1 = nn.Conv3d(1, n_planes, (7, 3, 3), padding=(1, 0, 0))
        # the number of kernels in the second convolution layer is set to be
        # twice as many as that in the first convolution layer
        self.conv2 = nn.Conv3d(n_planes, 2 * n_planes, (3, 3, 3), padding=(1, 0, 0))
        # self.dropout = nn.Dropout(p=0.5)
        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)
        self.reg = nn.Linear(self.features_size, input_channels)

        self.apply(self.weight_init)
        self.aux_loss_weight = 1

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = self.conv1(x)
            x = self.conv2(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.features_size)
        # x = self.dropout(x)
        x_class = self.fc(x)
        x_rec = self.reg(x)
        return x_class, x_rec

class zhoupatchdependency(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            init.constant_(m.bias.data, 0)

    def __init__(self, input_channels, n_classes, n_planes=16, patch_size=9):
        super(zhoupatchdependency, self).__init__()
        self.input_channels = input_channels
        self.n_planes = n_planes
        self.patch_size = patch_size
        # The proposed 3D-CNN model has two 3D convolution layers (C1 and C2)
        # and a fully-connected layer (F1)
        # we fix the spatial size of the 3D convolution kernels to 3 × 3
        # while only slightly varying the spectral depth of the kernels
        # for the Pavia University and Indian Pines scenes, those in C1 and C2
        # were set to seven and three, respectively
        #ouput of DRConv into 2rd Conv.
        self.conv2d1 = nn.Conv2d(in_channels=input_channels,out_channels=n_planes,kernel_size=3,stride=3)
        self.conv1 = nn.Conv3d(1, n_planes, (7, 3, 3), padding=(1, 0, 0),stride=(3,3,3))
        # the number of kernels in the second convolution layer is set to be
        # twice as many as that in the first convolution layer
        self.conv2d2 = nn.Conv2d(in_channels=n_planes, out_channels=n_planes*2,kernel_size=3)
        self.conv2 = nn.Conv3d(n_planes, 2 * n_planes,(3, 3, 3), padding=(1, 0, 0))
        self.dropout = nn.Dropout(p=0.5)
        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)
        self.softmax = nn.Softmax()
        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))

            x = self.conv2d1(x)
            x = self.conv2d2(x)
            _, t, c, w, h = x.size()
            print(t,c,w,h)
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv2d1(x))
        x = F.relu(self.conv2d2(x))
        x = x.view(-1, self.features_size)
        # x = self.dropout(x)
        x = self.fc(x)

        return x

class HeEtAl(nn.Module):
    """
    MULTI-SCALE 3D DEEP CONVOLUTIONAL NEURAL NETWORK FOR HYPERSPECTRAL
    IMAGE CLASSIFICATION
    Mingyi He, Bo Li, Huahui Chen
    IEEE International Conference on Image Processing (ICIP) 2017
    https://ieeexplore.ieee.org/document/8297014/
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_uniform(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=11):
        super(HeEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        self.conv1 = nn.Conv3d(1, 16, (11, 3, 3), stride=(3, 1, 1))
        self.conv2_1 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0, 0, 0))
        self.conv2_2 = nn.Conv3d(16, 16, (3, 1, 1), padding=(1, 0, 0))
        self.conv2_3 = nn.Conv3d(16, 16, (5, 1, 1), padding=(2, 0, 0))
        self.conv2_4 = nn.Conv3d(16, 16, (11, 1, 1), padding=(5, 0, 0))
        self.conv3_1 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0, 0, 0))
        self.conv3_2 = nn.Conv3d(16, 16, (3, 1, 1), padding=(1, 0, 0))
        self.conv3_3 = nn.Conv3d(16, 16, (5, 1, 1), padding=(2, 0, 0))
        self.conv3_4 = nn.Conv3d(16, 16, (11, 1, 1), padding=(5, 0, 0))
        self.conv4 = nn.Conv3d(16, 16, (3, 2, 2))
        self.pooling = nn.MaxPool2d((3, 2, 2), stride=(3, 2, 2))
        # the ratio of dropout is 0.6 in our experiments
        # self.dropout = nn.Dropout(p=0.6)

        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)
        self.reg = nn.Linear(self.features_size, input_channels)

        self.apply(self.weight_init)
        self.aux_loss_weight = 1

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = self.conv1(x)
            x2_1 = self.conv2_1(x)
            x2_2 = self.conv2_2(x)
            x2_3 = self.conv2_3(x)
            x2_4 = self.conv2_4(x)
            x = x2_1 + x2_2 + x2_3 + x2_4
            x3_1 = self.conv3_1(x)
            x3_2 = self.conv3_2(x)
            x3_3 = self.conv3_3(x)
            x3_4 = self.conv3_4(x)
            x = x3_1 + x3_2 + x3_3 + x3_4
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x2_1 = self.conv2_1(x)
        x2_2 = self.conv2_2(x)
        x2_3 = self.conv2_3(x)
        x2_4 = self.conv2_4(x)
        x = x2_1 + x2_2 + x2_3 + x2_4
        x = F.relu(x)
        x3_1 = self.conv3_1(x)
        x3_2 = self.conv3_2(x)
        x3_3 = self.conv3_3(x)
        x3_4 = self.conv3_4(x)
        x = x3_1 + x3_2 + x3_3 + x3_4
        x = F.relu(x)
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.features_size)
        # x = self.dropout(x)
        x_class = self.fc(x)
        x_reg = self.reg(x)
        return x_class, x_reg

class LuoEtAl(nn.Module):
    """
    HSI-CNN: A Novel Convolution Neural Network for Hyperspectral Image
    Yanan Luo, Jie Zou, Chengfei Yao, Tao Li, Gang Bai
    International Conference on Pattern Recognition 2018
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            init.kaiming_uniform_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=3, n_planes=90):
        super(LuoEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.n_planes = n_planes

        # the 8-neighbor pixels [...] are fed into the Conv1 convolved by n1 kernels
        # and s1 stride. Conv1 results are feature vectors each with height of and
        # the width is 1. After reshape layer, the feature vectors becomes an image-like 
        # 2-dimension data.
        # Conv2 has 64 kernels size of 3x3, with stride s2.
        # After that, the 64 results are drawn into a vector as the input of the fully 
        # connected layer FC1 which has n4 nodes.
        # In the four datasets, the kernel height nk1 is 24 and stride s1, s2 is 9 and 1
        self.conv1 = nn.Conv3d(1, 90, (24, 3, 3), padding=0, stride=(9,1,1))
        self.conv2 = nn.Conv2d(1, 64, (3, 3), stride=(1, 1))

        self.features_size = self._get_final_flattened_size()

        self.fc1 = nn.Linear(self.features_size, 1024)
        self.fc2 = nn.Linear(1024, n_classes)
        self.apply(self.weight_init)
        self.softmax = nn.Softmax()
        self.aux_loss_weight = 1

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.conv1(x)
            b = x.size(0)
            x = x.view(b, 1, -1, self.n_planes)
            x = self.conv2(x)
            _, c, w, h = x.size()
        return c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        b = x.size(0)
        x = x.view(b, 1, -1, self.n_planes)
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.features_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = self.softmax(x)
        # plt.plot(x[0, :].cpu().detach().numpy(), linewidth=2.5)
        # plt.xticks(fontsize=35)
        # plt.yticks(fontsize=35)
        # bwith = 2  # 边框宽度设置为2
        # TK = plt.gca()  # 获取边框
        # TK.spines['bottom'].set_linewidth(bwith)  # 图框下边
        # TK.spines['left'].set_linewidth(bwith)  # 图框左边
        # TK.spines['top'].set_linewidth(bwith)  # 图框上边
        # TK.spines['right'].set_linewidth(bwith)  # 图框右边
        # plt.show()
        return x

class SharmaEtAl(nn.Module):
    """
    HYPERSPECTRAL CNN FOR IMAGE CLASSIFICATION & BAND SELECTION, WITH APPLICATION
    TO FACE RECOGNITION
    Vivek Sharma, Ali Diba, Tinne Tuytelaars, Luc Van Gool
    Technical Report, KU Leuven/ETH Zürich
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv3d)):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=64):
        super(SharmaEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        # An input image of size 263x263 pixels is fed to conv1
        # with 96 kernels of size 6x6x96 with a stride of 2 pixels
        self.conv1 = nn.Conv3d(1, 96, (input_channels, 6, 6), stride=(1,2,2))
        self.conv1_bn = nn.BatchNorm3d(96)
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        #  256 kernels of size 3x3x256 with a stride of 2 pixels
        self.conv2 = nn.Conv3d(1, 256, (96, 3, 3), stride=(1,2,2))
        self.conv2_bn = nn.BatchNorm3d(256)
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        # 512 kernels of size 3x3x512 with a stride of 1 pixel
        self.conv3 = nn.Conv3d(1, 512, (256, 3, 3), stride=(1,1,1))
        # Considering those large kernel values, I assume they actually merge the
        # 3D tensors at each step

        self.features_size = self._get_final_flattened_size()

        # The fc1 has 1024 outputs, where dropout was applied after
        # fc1 with a rate of 0.5
        self.fc1 = nn.Linear(self.features_size, 1024)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, n_classes)

        self.apply(self.weight_init)
        self.aux_loss_weight = 1

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = F.relu(self.conv1_bn(self.conv1(x)))
            x = self.pool1(x)
            print(x.size())
            b, t, c, w, h = x.size()
            x = x.view(b, 1, t*c, w, h)
            x = F.relu(self.conv2_bn(self.conv2(x)))
            x = self.pool2(x)
            print(x.size())
            b, t, c, w, h = x.size()
            x = x.view(b, 1, t*c, w, h)
            x = F.relu(self.conv3(x))
            print(x.size())
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = self.pool1(x)
        b, t, c, w, h = x.size()
        x = x.view(b, 1, t*c, w, h)
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = self.pool2(x)
        b, t, c, w, h = x.size()
        x = x.view(b, 1, t*c, w, h)
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.features_size)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class LiuEtAl(nn.Module):
    """
    A semi-supervised convolutional neural network for hyperspectral image classification
    Bing Liu, Xuchu Yu, Pengqiang Zhang, Xiong Tan, Anzhu Yu, Zhixiang Xue
    Remote Sensing Letters, 2017
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=9):
        super(LiuEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.aux_loss_weight = 1

        # "W1 is a 3x3xB1 kernel [...] B1 is the number of the output bands for the convolutional
        # "and pooling layer" -> actually 3x3 2D convolutions with B1 outputs
        # "the value of B1 is set to be 80"
        self.conv1 = nn.Conv2d(input_channels, 80, (3, 3))
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv1_bn = nn.BatchNorm2d(80)

        self.features_sizes = self._get_sizes()

        self.fc_enc = nn.Linear(self.features_sizes[2], n_classes)

        # Decoder
        self.fc1_dec = nn.Linear(self.features_sizes[2], self.features_sizes[2])
        self.fc1_dec_bn = nn.BatchNorm1d(self.features_sizes[2])
        self.fc2_dec = nn.Linear(self.features_sizes[2], self.features_sizes[1])
        self.fc2_dec_bn = nn.BatchNorm1d(self.features_sizes[1])
        self.fc3_dec = nn.Linear(self.features_sizes[1], self.features_sizes[0])
        self.fc3_dec_bn = nn.BatchNorm1d(self.features_sizes[0])
        self.fc4_dec = nn.Linear(self.features_sizes[0], input_channels)

        self.apply(self.weight_init)
        self.aux_loss_weight = 1

    def _get_sizes(self):
        x = torch.zeros((1, self.input_channels, self.patch_size, self.patch_size))
        x = F.relu(self.conv1_bn(self.conv1(x)))
        _, c, w, h = x.size()
        size0 = c * w * h

        x = self.pool1(x)
        _, c, w, h = x.size()
        size1 = c * w * h

        x = self.conv1_bn(x)
        _, c, w, h = x.size()
        size2 = c * w * h

        return size0, size1, size2

    def forward(self, x):
        x = x.squeeze(1)
        x_conv1 = self.conv1_bn(self.conv1(x))
        x = x_conv1
        x_pool1 = self.pool1(x)
        x = x_pool1
        x_enc = F.relu(x).view(-1, self.features_sizes[2])
        x = x_enc

        x_classif = self.fc_enc(x)

        # x = F.relu(self.fc1_dec_bn(self.fc1_dec(x) + x_enc))
        x = F.relu(self.fc1_dec(x))
        x = F.relu(
            self.fc2_dec_bn(self.fc2_dec(x) + x_pool1.view(-1, self.features_sizes[1]))
        )
        x = F.relu(
            self.fc3_dec_bn(self.fc3_dec(x) + x_conv1.view(-1, self.features_sizes[0]))
        )
        x = self.fc4_dec(x)
        return x_classif, x

class BoulchEtAl(nn.Module):
    """
    Autoencodeurs pour la visualisation d'images hyperspectrales
    A.Boulch, N. Audebert, D. Dubucq
    GRETSI 2017
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, planes=16):
        super(BoulchEtAl, self).__init__()
        self.input_channels = input_channels
        # self.aux_loss_weight = 0.1
        self.aux_loss_weight = 1

        encoder_modules = []
        n = input_channels
        with torch.no_grad():
            x = torch.zeros((10, 1, self.input_channels))
            print(x.size())
            while(n > 1):
                print("---------- {} ---------".format(n))
                if n == input_channels:
                    p1, p2 = 1, 2 * planes
                elif n == input_channels // 2:
                    p1, p2 = 2 * planes, planes
                else:
                    p1, p2 = planes, planes
                encoder_modules.append(nn.Conv1d(p1, p2, 3, padding=1))
                x = encoder_modules[-1](x)
                print(x.size())
                encoder_modules.append(nn.MaxPool1d(2))
                x = encoder_modules[-1](x)
                print(x.size())
                encoder_modules.append(nn.ReLU(inplace=True))
                x = encoder_modules[-1](x)
                print(x.size())
                encoder_modules.append(nn.BatchNorm1d(p2))
                x = encoder_modules[-1](x)
                print(x.size())
                n = n // 2

            encoder_modules.append(nn.Conv1d(planes, 3, 3, padding=1))
        encoder_modules.append(nn.Tanh())
        self.encoder = nn.Sequential(*encoder_modules)
        self.features_sizes = self._get_sizes()

        self.classifier = nn.Linear(self.features_sizes, n_classes)
        self.regressor = nn.Linear(self.features_sizes, input_channels)
        self.apply(self.weight_init)

    def _get_sizes(self):
        with torch.no_grad():
            x = torch.zeros((10, 1, self.input_channels))
            x = self.encoder(x)
            _, c, w = x.size()
        return c*w

    def forward(self, x):
        print('x in',x.shape)
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = x.view(-1, self.features_sizes)
        x_classif = self.classifier(x)
        x = self.regressor(x)
        print('x class and x', x_classif.shape, x.shape) #(batch num class), (batch, channel)=input size
        return x_classif, x

class MouEtAl(nn.Module):
    """
    Deep recurrent neural networks for hyperspectral image classification
    Lichao Mou, Pedram Ghamisi, Xiao Xang Zhu
    https://ieeexplore.ieee.org/document/7914752/
    """

    @staticmethod
    def weight_init(m):
        # All weight matrices in our RNN and bias vectors are initialized with a uniform distribution, and the values of these weight matrices and bias vectors are initialized in the range [−0.1,0.1]
        if isinstance(m, (nn.Linear, nn.GRU)):
            init.uniform_(m.weight.data, -0.1, 0.1)
            init.uniform_(m.bias.data, -0.1, 0.1)

    def __init__(self, input_channels, n_classes):
        # The proposed network model uses a single recurrent layer that adopts our modified GRUs of size 64 with sigmoid gate activation and PRetanh activation functions for hidden representations
        super(MouEtAl, self).__init__()
        self.input_channels = input_channels
        self.gru = nn.GRU(1, 64, 1, bidirectional=False)  # TODO: try to change this ?
        self.gru_bn = nn.BatchNorm1d(64 * input_channels)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(64 * input_channels, n_classes)
        self.aux_loss_weight = 1

    def forward(self, x):
        print('x',x.shape)
        # x = x.squeeze()
        x = x.unsqueeze(0)
        # x is in 1, N, C but we expect C, N, 1 for GRU layer
        x = x.permute(2, 1, 0)
        x = self.gru(x)[0]
        # x is in C, N, 64, we permute back
        x = x.permute(1, 2, 0).contiguous()
        x = x.view(x.size(0), -1)
        x = self.gru_bn(x)
        x = self.tanh(x)
        x = self.fc(x)
        return

class zhouSpectralAttention(nn.Module):

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.GRU)):
            init.uniform_(m.weight.data)
            init.uniform_(m.bias.data)

    def __init__(self, input_channels, n_classes, embed_fea = 64, num_layer=1):
        super(zhouSpectralAttention, self).__init__()
        self.aux_loss_weight = 1
        self.input_channels = input_channels
        self.gru = nn.GRU(1, embed_fea, num_layer,
                            bidirectional=False)  # TODO: try to change this ? #之前是（1,64,1） pavia——100用100的feature hidden。 indianpine用200
        self.gru_1 = nn.GRU(1,embed_fea, num_layer, bidirectional= False)  # TODO: try to change this ? #之前是（1,64,1） pavia——100用100的feature hidden。 indianpine用200
        self.gru_2 = nn.GRU(1, embed_fea, num_layer,
                          bidirectional=False)  # TODO: try to change this ? #之前是（1,64,1） pavia——100用100的feature hidden。 indianpine用200
        self.gru_3 = nn.GRU(1, embed_fea, num_layer,
                          bidirectional=False)  # TODO: try to change this ? #之前是（1,64,1） pavia——100用100的feature hidden。 indianpine用200
        self.gru_4 = nn.GRU(1, embed_fea, num_layer,
                          bidirectional=False)  # TODO: try to change this ? #之前是（1,64,1） pavia——100用100的feature hidden。 indianpine用200
        self.gru_5 = nn.GRU(1, embed_fea, num_layer,
                          bidirectional=False)  # TODO: try to change this ? #之前是（1,64,1） pavia——100用100的feature hidden。 indianpine用200
        self.gru_6 = nn.GRU(1, embed_fea, num_layer,
                          bidirectional=False)  # TODO: try to change this ? #之前是（1,64,1） pavia——100用100的feature hidden。 indianpine用200
        self.gru_7 = nn.GRU(1, embed_fea, num_layer,
                          bidirectional=False)  # TODO: try to change this ? #之前是（1,64,1） pavia——100用100的feature hidden。 indianpine用200
        self.gru_8 = nn.GRU(1, embed_fea, num_layer,
                          bidirectional=False)  # TODO: try to change this ? #之前是（1,64,1） pavia——100用100的feature hidden。 indianpine用200
        self.gru_9 = nn.GRU(1, embed_fea, num_layer,
                          bidirectional=False)  # TODO: try to change this ? #之前是（1,64,1） pavia——100用100的feature hidden。 indianpine用200
        self.gru_10 = nn.GRU(1, embed_fea, num_layer,
                          bidirectional=False)  # TODO: try to change this ? #之前是（1,64,1） pavia——100用100的feature hidden。 indianpine用200
        # self.gru2 = nn.GRU(1,1,1)
        self.gru3 = nn.GRU(embed_fea,embed_fea,embed_fea, bidirectional= False)#之前是（64,64，1）  pavia——100用100的feature hidden。 indianpine用200如果用双向改成100
        # self.gru4 = nn.GRU(1,200,1)#之前是（1,64,1） pavia——100用100的feature hidden。 indianpine用200
        self.gru_bn = nn.BatchNorm1d(embed_fea )#之前是（64*64）根据 记得根据数据集更改input——channel pavia——100用100的feature hidden。 indianpine用200
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(embed_fea , n_classes) #之前是（64*64,）记得根据数据集更改input——channel pavia——100用100的feature hidden。 indianpine用200
        self.regressor = nn.Linear(embed_fea,input_channels)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(p=0.5)
        # self.model2 = HuEtAl(input_channels=input_channels,n_classes=n_classes)
        self.aux_loss_weight = 1

    def forward(self, x):
        print('1',x.shape)
        # pre2 = self.model2(x)
        # x = x.squeeze()
        print('2',x.shape)
        x = x.unsqueeze(0)
        x = x.unsqueeze(1)
        print('3',x.shape)
        # x is in 1, N, C but we expect C, N, 1 for GRU layer
        x = x.permute(2, 1, 0)
        # plt.subplot(4,1,1)
        # plt.plot(x[:, 1, :].cpu().detach().numpy(),linewidth='2.5')
        print('4',x.shape)
        x1_10,x11_20,x21_30,x31_40,x41_50,x51_60,x61_70,x71_80,x81_90,x91_100 = x.chunk(10,0)
        # plt.subplot(3, 3, 1)
        # plt.plot(x1_10[:,1, :].cpu().detach().numpy())
        # plt.subplot(3, 3, 2)
        # plt.plot(x11_20[:,1,:].cpu().detach().numpy())
        # plt.subplot(3, 3, 3)
        # plt.plot(x21_30[:,1, :].cpu().detach().numpy())
        x1_10 = self.gru_1(x1_10)[0]
        print('x1_10',x1_10.shape)
        x1_10_l = x1_10[-1] #64
        x1_10_l = torch.unsqueeze(x1_10_l, 0)
        # plt.subplot(3,3,4)
        # plt.plot(x1_10[:, 1, :].cpu().detach().numpy())
        # plt.plot(x1_10_l[1,:].cpu().detach().numpy())
        # plt.subplot(3,3,7)
        # plt.plot(x1_10_l.cpu().detach().numpy())
        x11_20 = self.gru_2(x11_20)[0]
        x11_20_l = x11_20[-1]
        x11_20_l = torch.unsqueeze(x11_20_l, 0)
        print('x_11_20_l', x11_20_l.shape) #(1, batch, 64)
        # plt.subplot(3,3,5)
        # plt.plot(x11_20[:, 1, :].cpu().detach().numpy())
        x21_30 = self.gru_3(x21_30)[0]
        x21_30_l = x21_30[-1]
        x21_30_l = torch.unsqueeze(x21_30_l, 0)
        # plt.subplot(3,3,6)
        # plt.plot(x21_30[:, 1, :].cpu().detach().numpy())
        x31_40 = self.gru_4(x31_40)[0]
        x31_40_l = x31_40[-1]
        x31_40_l = torch.unsqueeze(x31_40_l,0)

        x41_50 = self.gru_5(x41_50)[0]
        x41_50_l = x41_50[-1]
        x41_50_l = torch.unsqueeze(x41_50_l, 0)

        x51_60 = self.gru_6(x51_60)[0]
        x51_60_l = x51_60[-1]
        x51_60_l = torch.unsqueeze(x51_60_l, 0)

        x61_70 = self.gru_7(x61_70)[0]
        x61_70_l = x61_70[-1]
        x61_70_l = torch.unsqueeze(x61_70_l, 0)

        x71_80 = self.gru_8(x71_80)[0]
        x71_80_l = x71_80[-1]
        x71_80_l = torch.unsqueeze(x71_80_l, 0)

        x81_90 = self.gru_9(x81_90)[0]
        x81_90_l = x81_90[-1]
        x81_90_l = torch.unsqueeze(x81_90_l,0)

        x91_100 = self.gru_10(x91_100)[0]
        x91_100_l = x91_100[-1]
        x91_100_l = torch.unsqueeze(x91_100_l,0)  #size 1,batch,feature
        print('x_91_100_l',x91_100_l.shape)
        # x91_100_l = x91_100_l.expand(x91_100_l.shape[0],10) 这个可以把seq扩张成matrix

        x_cat = torch.cat([x1_10_l,x11_20_l,x21_30_l,x31_40_l,x41_50_l,x51_60_l,x61_70_l,x71_80_l,x81_90_l,x91_100_l],dim=0)  #size 10,batch,feature
        print('x_cat', x_cat.shape) # 10, batch, 64
        x_cat = self.gru3(x_cat)[0]
        print('x_cat', x_cat.shape)# 10, batch, 64
        x_cat_l = torch.avg_pool1d(rearrange(x_cat,'l n c -> n c l'),kernel_size=10)
        # x_cat_l = x_cat[-1] #size  (batch,feature=64)
        print('x_cat_l[-1]',x_cat_l.shape) #batch,64
        x_cat_l = self.relu(x_cat_l)
        # plt.subplot(4, 1, 3)
        # plt.plot(x_cat_l[1, :].cpu().detach().numpy(),linewidth='2.5')
        x = self.gru(x)[0]
        print('x:',x.shape) # l n c
        x_l = torch.avg_pool1d(rearrange(x, 'l n c -> n c l'), kernel_size=self.input_channels)
        # x_l = x[-1] #size batch,feature=64
        # plt.subplot(4, 1, 2)
        # plt.plot(x_l[1, :].cpu().detach().numpy(),linewidth='2.5')
        print('x_l[-1]',x_l.shape) # 1 100 64
        x_new = x_l * x_cat_l  #改成+号试试
        print('x_new',x_new.shape)
        # plt.subplot(4, 1, 4)
        # plt.plot(x_new[1, :].cpu().detach().numpy(),linewidth='5')
        # x_new = torch.unsqueeze(x_new,0)
        print('x_new',x_new.shape) # 1 , batch, 64
        # x_new = x_new.permute(2,1,0) #feature 变为 seq
        print('x_new',x_new.shape) #size: seq,batch,1
        x = x_new
        # plt.subplot(4,1,2)
        # plt.plot(x_cat[:, 1, :].cpu().detach().numpy())
        # x_cat = self.gru2(x_cat)[0]
        # x_cat = self.tanh(x_cat)
        # plt.subplot(4,1,3)
        # plt.plot(x_cat[:, 1, :].cpu().detach().numpy())
        # plt.subplot(3, 3, 8)
        # plt.plot(x_cat[:, 1, :].cpu().detach().numpy())
        # print('x1_10', x1_10.shape)
        # print('x91_100', x91_100.shape)
        # plt.subplot(142)
        # plt.plot(x_cat[:,1,:].cpu().detach().numpy())
        # x_cat = self.gru4(x_cat)[0]
        # plt.subplot(143)
        # plt.plot(x_cat[:, 1, :].cpu().detach().numpy())
        # x_cat_1 = x_cat[-1]
        # # x_cat_1 = self.relu(x_cat_1)
        # plt.subplot(144)
        # plt.plot(x_cat_1.cpu().detach().numpy())
        # x = x + x_cat
        # x = x * x_cat
        # plt.subplot(4,1,4)
        # plt.plot(x[:,1,:].cpu().detach().numpy())
        # plt.subplot(3, 3, 9)
        # plt.plot(x[:, 1, :].cpu().detach().numpy())
        # plt.imshow(x[:,1,:].cpu().detach().numpy())
        # x = self.gru(x_new)[0]
        # plt.subplot(144)
        # plt.plot(x[:, 1, :].cpu().detach().numpy())
        # plt.show()
        # x = self.gru3(x)[0]
        print('5',x.shape)
        # x is in C, N, 64, we permute back
        # x = x.permute(1, 2, 0).contiguous()
        print('6',x.shape)
        x = x.view(x.size(0), -1)
        print('7',x.shape)
        # x = nn.BatchNorm1d(x.shape[1])
        x = self.gru_bn(x)
        # x = self.tanh(x)
        x = self.relu(x)
        x_rec = self.regressor(x)
        # x = nn.Linear(x.shape[1],10)
        # x = self.dropout(x)
        x_class = self.fc(x)
        # plt.grid(linewidth = 0.5, color = 'black' )
        # plt.title('Visualiazations in the block', fontdict={'size':40})
        # plt.legend(['x','ff','fx','fnew'], prop={'size':40}, fontsize = 'large')
        # plt.xlabel('Feature size', fontdict={'size':40}, fontweight = 'bold')
        # plt.ylabel('Feature value',fontdict={'size':40})
        # plt.xticks(fontsize=35)
        # plt.yticks(fontsize=35)
        # bwith = 2  # 边框宽度设置为2
        # TK = plt.gca()  # 获取边框
        # TK.spines['bottom'].set_linewidth(bwith)  # 图框下边
        # TK.spines['left'].set_linewidth(bwith)  # 图框左边
        # TK.spines['top'].set_linewidth(bwith)  # 图框上边
        # TK.spines['right'].set_linewidth(bwith)  # 图框右边
        # plt.show()
        # print('after fc', x.shape)
        return x_class, x_rec

class zhou16dRNN(nn.Module):
    @staticmethod
    def weight_init(m):
        # All weight matrices in our RNN and bias vectors are initialized with a uniform distribution, and the values of these weight matrices and bias vectors are initialized in the range [−0.1,0.1]
        if isinstance(m, (nn.Linear, nn.GRU, nn.Conv3d)):
            init.kaiming_uniform_(m.weight.data)
            init.kaiming_uniform_(m.bias.data)

    def __init__(self, input_channels, n_classes, patch_size=3):
        # The proposed network model uses a single recurrent layer that adopts our modified GRUs of size 64 with sigmoid gate activation and PRetanh activation functions for hidden representations
        super(zhou16dRNN, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.gru = nn.GRU(patch_size ** 2, patch_size ** 2, 1, bidirectional=False,
                          batch_first=False)  # TODO: try to change this ?
        self.gru_2_1 = nn.GRU(input_channels, patch_size**2, 1, bidirectional=False)
        self.gru_2_2 = nn.GRU(input_channels, patch_size**2, 1, bidirectional=False)
        self.gru_2_3 = nn.GRU(input_channels, patch_size**2, 1, bidirectional=False)
        self.gru_2_4 = nn.GRU(input_channels, patch_size**2, 1, bidirectional=False)
        self.gru_2_5 = nn.GRU(input_channels, patch_size**2, 1, bidirectional=False)
        self.gru_2_6 = nn.GRU(input_channels, patch_size**2, 1, bidirectional=False)
        self.gru_2_7 = nn.GRU(input_channels, patch_size**2, 1, bidirectional=False)
        self.gru_2_8 = nn.GRU(input_channels, patch_size**2, 1, bidirectional=False)
        self.gru_2_9 = nn.GRU(input_channels, patch_size**2, 1, bidirectional=False)
        self.gru_2_10 = nn.GRU(input_channels, patch_size**2, 1, bidirectional=False)
        self.gru_2_11 = nn.GRU(input_channels, patch_size**2, 1, bidirectional=False)
        self.gru_2_12 = nn.GRU(input_channels, patch_size**2, 1, bidirectional=False)
        self.gru_2_13 = nn.GRU(input_channels, patch_size**2, 1, bidirectional=False)
        self.gru_2_14 = nn.GRU(input_channels, patch_size**2, 1, bidirectional=False)
        self.gru_2_15 = nn.GRU(input_channels, patch_size**2, 1, bidirectional=False)
        self.gru_2_16 = nn.GRU(input_channels, patch_size**2, 1, bidirectional=False)
        self.gru_3 = nn.GRU(input_channels, patch_size ** 2, 1, bidirectional=True)
        self.gru_bn = nn.BatchNorm1d(patch_size ** 2 * input_channels)
        self.gru_bn_2 = nn.BatchNorm1d((patch_size**2) * (patch_size**2))
        self.gru_bn_3 = nn.BatchNorm1d((patch_size ** 2) * (patch_size ** 2) * 2)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(patch_size ** 2 * input_channels, n_classes)
        self.fc_2 = nn.Linear((patch_size ** 2) * (patch_size**2), n_classes)
        self.fc_3 = nn.Linear((patch_size ** 2) * (patch_size ** 2) * 2, n_classes)
    def forward(self, x):  # 初始是第1方向
        # print("x",x.shape)
        # x1_1 = self.conv_1x1(x)
        # x3_3 = self.conv_3x3(x)
        # x5_5 = self.conv_5_5(x)
        # x = torch.cat([x3_3, x5_5, x1_1], dim=1)

        x = x.squeeze()
        x1_1 = x
        # print("x1_1",x1_1.shape)
        l = x1_1.shape[0]
        for i in range(l):
            x1_1i = x1_1[i,:,:,:]

            y1_1r0 = torch.diagonal(x1_1i, 0, 2)
            y1_1r0 = y1_1r0.cpu()
            y1_1r0_2n = np.flip(y1_1r0.numpy(), axis=1).copy()
            y1_1r0_2t = torch.from_numpy(y1_1r0_2n).cuda()
            y1_1r0 = y1_1r0.cuda()

            y1_1r1 = torch.diagonal(x1_1i, 1, 2)
            y1_1r1 = y1_1r1.cpu()
            y1_1r1_2n = np.flip(y1_1r1.numpy(), axis=1).copy()
            y1_1r1_2t = torch.from_numpy(y1_1r1_2n).cuda()
            y1_1r1 = y1_1r1.cuda()

            y1_1r2 = torch.diagonal(x1_1i, 2, 2)
            y1_1r2 = y1_1r2.cpu()
            y1_1r2_2n = np.flip(y1_1r2.numpy(), axis=1).copy()
            y1_1r2_2t = torch.from_numpy(y1_1r2_2n).cuda()
            y1_1r2= y1_1r2.cuda()

            # y1_1r3 = torch.diagonal(x1_1i, 3, 2)
            # y1_1r3 = y1_1r3.cpu()
            # y1_1r3_2n = np.flip(y1_1r3.numpy(), axis=1).copy()
            # y1_1r3_2t = torch.from_numpy(y1_1r3_2n).cuda()
            # y1_1r3 = y1_1r3.cuda()

            # y1_1r4 = torch.diagonal(x1_1i, 4, 2)
            # y1_1r4 = y1_1r4.cpu()
            # y1_1r4_2n = np.flip(y1_1r4.numpy(), axis=1).copy()
            # y1_1r4_2t = torch.from_numpy(y1_1r4_2n).cuda()
            # y1_1r4 = y1_1r4.cuda()

            y1_1rf1 = torch.diagonal(x1_1i, -1, 2)
            y1_1rf1 = y1_1rf1.cpu()
            y1_1rf1_2n = np.flip(y1_1rf1.numpy(), axis=1).copy()
            y1_1rf1_2t = torch.from_numpy(y1_1rf1_2n).cuda()
            y1_1rf1 = y1_1rf1.cuda()

            y1_1rf2 = torch.diagonal(x1_1i, -2, 2)
            y1_1rf2 = y1_1rf2.cpu()
            y1_1rf2_2n = np.flip(y1_1rf2.numpy(), axis=1).copy()
            y1_1rf2_2t = torch.from_numpy(y1_1rf2_2n).cuda()
            y1_1rf2 = y1_1rf2.cuda()

            # y1_1rf3 = torch.diagonal(x1_1i, -3, 2)
            # y1_1rf3 = y1_1rf3.cpu()
            # y1_1rf3_2n = np.flip(y1_1rf3.numpy(), axis=1).copy()
            # y1_1rf3_2t = torch.from_numpy(y1_1rf3_2n).cuda()
            # y1_1rf3 = y1_1rf3.cuda()
            #
            # y1_1rf4 = torch.diagonal(x1_1i, -4, 2)
            # y1_1rf4 = y1_1rf4.cpu()
            # y1_1rf4_2n = np.flip(y1_1rf4.numpy(), axis=1).copy()
            # y1_1rf4_2t = torch.from_numpy(y1_1rf4_2n).cuda()
            # y1_1rf4 = y1_1rf4.cuda()

            # diagonal-1
            xd1s = torch.cat([y1_1r2, y1_1r1, y1_1r0, y1_1rf1, y1_1rf2], dim=1)
            # xd1s = torch.cat([y1_1r4, y1_1r3, y1_1r2, y1_1r1, y1_1r0, y1_1rf1, y1_1rf2, y1_1rf3, y1_1rf4], dim=1)
            # print('xd1', xd1.shape)
            # diagonal-2
            xd2s = torch.cat([ y1_1r2_2t, y1_1r1_2t, y1_1r0_2t, y1_1rf1_2t, y1_1rf2_2t], dim=1)
            #
            # xd2s = torch.cat([y1_1r4_2t, y1_1r3_2t, y1_1r2_2t, y1_1r1_2t, y1_1r0_2t, y1_1rf1_2t, y1_1rf2_2t, y1_1rf3_2t, y1_1rf4_2t], dim=1)
            # print('xd2', xd2.shape)

            x2_2dia = torch.rot90(x1_1i, 1, dims=[1, 2])

            y2_2r0 = torch.diagonal(x2_2dia, 0, 2)
            y2_2r0 = y2_2r0.cpu()
            y2_2r0_2n = np.flip(y2_2r0.numpy(), axis=1).copy()
            y2_2r0_2t = torch.from_numpy(y2_2r0_2n).cuda()
            y2_2r0 = y2_2r0.cuda()

            y2_2r1 = torch.diagonal(x2_2dia, 1, 2)
            y2_2r1 = y2_2r1.cpu()
            y2_2r1_2n = np.flip(y2_2r1.numpy(), axis=1).copy()
            y2_2r1_2t = torch.from_numpy(y2_2r1_2n).cuda()
            y2_2r1 = y2_2r1.cuda()

            y2_2r2 = torch.diagonal(x2_2dia, 2, 2)
            y2_2r2 = y2_2r2.cpu()
            y2_2r2_2n = np.flip(y2_2r2.numpy(), axis=1).copy()
            y2_2r2_2t = torch.from_numpy(y2_2r2_2n).cuda()
            y2_2r2 = y2_2r2.cuda()

            # y2_2r3 = torch.diagonal(x2_2dia, 3, 2)
            # y2_2r3 = y2_2r3.cpu()
            # y2_2r3_2n = np.flip(y2_2r3.numpy(), axis=1).copy()
            # y2_2r3_2t = torch.from_numpy(y2_2r3_2n).cuda()
            # y2_2r3 = y2_2r3.cuda()
            #
            # y2_2r4 = torch.diagonal(x2_2dia, 4, 2)
            # y2_2r4 = y2_2r4.cpu()
            # y2_2r4_2n = np.flip(y2_2r4.numpy(), axis=1).copy()
            # y2_2r4_2t = torch.from_numpy(y2_2r4_2n).cuda()
            # y2_2r4 = y2_2r4.cuda()

            y2_2rf1 = torch.diagonal(x2_2dia, -1, 2)
            y2_2rf1 = y2_2rf1.cpu()
            y2_2rf1_2n = np.flip(y2_2rf1.numpy(), axis=1).copy()
            y2_2rf1_2t = torch.from_numpy(y2_2rf1_2n).cuda()
            y2_2rf1 = y2_2rf1.cuda()

            y2_2rf2 = torch.diagonal(x2_2dia, -2, 2)
            y2_2rf2 = y2_2rf2.cpu()
            y2_2rf2_2n = np.flip(y2_2rf2.numpy(), axis=1).copy()
            y2_2rf2_2t = torch.from_numpy(y2_2rf2_2n).cuda()
            y2_2rf2 = y2_2rf2.cuda()

            # y2_2rf3 = torch.diagonal(x2_2dia, -3, 2)
            # y2_2rf3 = y2_2rf3.cpu()
            # y2_2rf3_2n = np.flip(y2_2rf3.numpy(), axis=1).copy()
            # y2_2rf3_2t = torch.from_numpy(y2_2rf3_2n).cuda()
            # y2_2rf3 = y2_2rf3.cuda()
            #
            # y2_2rf4 = torch.diagonal(x2_2dia, -4, 2)
            # y2_2rf4 = y2_2rf4.cpu()
            # y2_2rf4_2n = np.flip(y2_2rf4.numpy(), axis=1).copy()
            # y2_2rf4_2t = torch.from_numpy(y2_2rf4_2n).cuda()
            # y2_2rf4 = y2_2rf4.cuda()

            # diagonal-3
            xd3s = torch.cat([ y2_2r2, y2_2r1, y2_2r0, y2_2rf1, y2_2rf2], dim=1)
            # xd3s = torch.cat([y2_2r4, y2_2r3, y2_2r2, y2_2r1, y2_2r0, y2_2rf1, y2_2rf2, y2_2rf3, y2_2rf4], dim=1)
            # print('xd3', xd3.shape)
            # diagonal-4
            xd4s = torch.cat([y2_2r2_2t, y2_2r1_2t, y2_2r0_2t, y2_2rf1_2t, y2_2rf2_2t],dim=1)
            # xd4s = torch.cat([y2_2r4_2t, y2_2r3_2t, y2_2r2_2t, y2_2r1_2t, y2_2r0_2t, y2_2rf1_2t, y2_2rf2_2t, y2_2rf3_2t, y2_2rf4_2t], dim=1)
            # print('xd4', xd4.shape)

            x3_3dia = torch.transpose(x1_1i, 1, 2)

            y3_3r0 = torch.diagonal(x3_3dia, 0, 2)
            y3_3r0 = y3_3r0.cpu()
            y3_3r0_2n = np.flip(y3_3r0.numpy(), axis=1).copy()
            y3_3r0_2t = torch.from_numpy(y3_3r0_2n).cuda()
            y3_3r0 = y3_3r0.cuda()

            y3_3r1 = torch.diagonal(x3_3dia, 1, 2)
            y3_3r1 = y3_3r1.cpu()
            y3_3r1_2n = np.flip(y3_3r1.numpy(), axis=1).copy()
            y3_3r1_2t = torch.from_numpy(y3_3r1_2n).cuda()
            y3_3r1 = y3_3r1.cuda()

            y3_3r2 = torch.diagonal(x3_3dia, 2, 2)
            y3_3r2 = y3_3r2.cpu()
            y3_3r2_2n = np.flip(y3_3r2.numpy(), axis=1).copy()
            y3_3r2_2t = torch.from_numpy(y3_3r2_2n).cuda()
            y3_3r2 = y3_3r2.cuda()
            #
            # y3_3r3 = torch.diagonal(x3_3dia, 3, 2)
            # y3_3r3 = y3_3r3.cpu()
            # y3_3r3_2n = np.flip(y3_3r3.numpy(), axis=1).copy()
            # y3_3r3_2t = torch.from_numpy(y3_3r3_2n).cuda()
            # y3_3r3 = y3_3r3.cuda()
            #
            # y3_3r4 = torch.diagonal(x3_3dia, 4, 2)
            # y3_3r4 = y3_3r4.cpu()
            # y3_3r4_2n = np.flip(y3_3r4.numpy(), axis=1).copy()
            # y3_3r4_2t = torch.from_numpy(y3_3r4_2n).cuda()
            # y3_3r4 = y3_3r4.cuda()

            y3_3rf1 = torch.diagonal(x3_3dia, -1, 2)
            y3_3rf1 = y3_3rf1.cpu()
            y3_3rf1_2n = np.flip(y3_3rf1.numpy(), axis=1).copy()
            y3_3rf1_2t = torch.from_numpy(y3_3rf1_2n).cuda()
            y3_3rf1 = y3_3rf1.cuda()

            y3_3rf2 = torch.diagonal(x3_3dia, -2, 2)
            y3_3rf2 = y3_3rf2.cpu()
            y3_3rf2_2n = np.flip(y3_3rf2.numpy(), axis=1).copy()
            y3_3rf2_2t = torch.from_numpy(y3_3rf2_2n).cuda()
            y3_3rf2 = y3_3rf2.cuda()

            # y3_3rf3 = torch.diagonal(x3_3dia, -3, 2)
            # y3_3rf3 = y3_3rf3.cpu()
            # y3_3rf3_2n = np.flip(y3_3rf3.numpy(), axis=1).copy()
            # y3_3rf3_2t = torch.from_numpy(y3_3rf3_2n).cuda()
            # y3_3rf3 = y3_3rf3.cuda()
            #
            # y3_3rf4 = torch.diagonal(x3_3dia, -4, 2)
            # y3_3rf4 = y3_3rf4.cpu()
            # y3_3rf4_2n = np.flip(y3_3rf4.numpy(), axis=1).copy()
            # y3_3rf4_2t = torch.from_numpy(y3_3rf4_2n).cuda()
            # y3_3rf4 = y3_3rf4.cuda()

            # diagonal-5
            xd5s = torch.cat([ y3_3r2, y3_3r1, y3_3r0, y3_3rf1, y3_3rf2], dim=1)
            # xd5s = torch.cat([y3_3r4, y3_3r3, y3_3r2, y3_3r1, y3_3r0, y3_3rf1, y3_3rf2, y3_3rf3, y3_3rf4], dim=1)
            # print('xd5', xd5.shape)
            # diagonal-6
            xd6s = torch.cat([ y3_3r2_2t, y3_3r1_2t, y3_3r0_2t, y3_3rf1_2t, y3_3rf2_2t],dim=1)
            # xd6s = torch.cat([y3_3r4_2t, y3_3r3_2t, y3_3r2_2t, y3_3r1_2t, y3_3r0_2t, y3_3rf1_2t, y3_3rf2_2t, y3_3rf3_2t, y3_3rf4_2t], dim=1)
            # print('xd6', xd6.shape)

            x4_4dia = torch.rot90(x3_3dia, 3, [1, 2])

            y4_4r0 = torch.diagonal(x4_4dia, 0, 2)
            y4_4r0 = y4_4r0.cpu()
            y4_4r0_2n = np.flip(y4_4r0.numpy(), axis=1).copy()
            y4_4r0_2t = torch.from_numpy(y4_4r0_2n).cuda()
            y4_4r0 = y4_4r0.cuda()

            y4_4r1 = torch.diagonal(x4_4dia, 1, 2)
            y4_4r1 = y4_4r1.cpu()
            y4_4r1_2n = np.flip(y4_4r1.numpy(), axis=1).copy()
            y4_4r1_2t = torch.from_numpy(y4_4r1_2n).cuda()
            y4_4r1 = y4_4r1.cuda()

            y4_4r2 = torch.diagonal(x4_4dia, 2, 2)
            y4_4r2 = y4_4r2.cpu()
            y4_4r2_2n = np.flip(y4_4r2.numpy(), axis=1).copy()
            y4_4r2_2t = torch.from_numpy(y4_4r2_2n).cuda()
            y4_4r2 = y4_4r2.cuda()

            # y4_4r3 = torch.diagonal(x4_4dia, 3, 2)
            # y4_4r3 = y4_4r3.cpu()
            # y4_4r3_2n = np.flip(y4_4r3.numpy(), axis=1).copy()
            # y4_4r3_2t = torch.from_numpy(y4_4r3_2n).cuda()
            # y4_4r3 = y4_4r3.cuda()
            #
            # y4_4r4 = torch.diagonal(x4_4dia, 4, 2)
            # y4_4r4 = y4_4r4.cpu()
            # y4_4r4_2n = np.flip(y4_4r4.numpy(), axis=1).copy()
            # y4_4r4_2t = torch.from_numpy(y4_4r4_2n).cuda()
            # y4_4r4 = y4_4r4.cuda()

            y4_4rf1 = torch.diagonal(x4_4dia, -1, 2)
            y4_4rf1 = y4_4rf1.cpu()
            y4_4rf1_2n = np.flip(y4_4rf1.numpy(), axis=1).copy()
            y4_4rf1_2t = torch.from_numpy(y4_4rf1_2n).cuda()
            y4_4rf1 = y4_4rf1.cuda()

            y4_4rf2 = torch.diagonal(x4_4dia, -2, 2)
            y4_4rf2 = y4_4rf2.cpu()
            y4_4rf2_2n = np.flip(y4_4rf2.numpy(), axis=1).copy()
            y4_4rf2_2t = torch.from_numpy(y4_4rf2_2n).cuda()
            y4_4rf2 = y4_4rf2.cuda()

            # y4_4rf3 = torch.diagonal(x4_4dia, -3, 2)
            # y4_4rf3 = y4_4rf3.cpu()
            # y4_4rf3_2n = np.flip(y4_4rf3.numpy(), axis=1).copy()
            # y4_4rf3_2t = torch.from_numpy(y4_4rf3_2n).cuda()
            # y4_4rf3 = y4_4rf3.cuda()
            #
            # y4_4rf4 = torch.diagonal(x4_4dia, -4, 2)
            # y4_4rf4 = y4_4rf4.cpu()
            # y4_4rf4_2n = np.flip(y4_4rf4.numpy(), axis=1).copy()
            # y4_4rf4_2t = torch.from_numpy(y4_4rf4_2n).cuda()
            # y4_4rf4 = y4_4rf4.cuda()

            # diagonal-7
            xd7s = torch.cat([y4_4r2, y4_4r1, y4_4r0, y4_4rf1, y4_4rf2], dim=1)
            # xd7s = torch.cat([y4_4r4, y4_4r3, y4_4r2, y4_4r1, y4_4r0, y4_4rf1, y4_4rf2, y4_4rf3, y4_4rf4], dim=1)
            # print('xd7', xd7.shape)
            # diagonal-8
            xd8s = torch.cat([ y4_4r2_2t, y4_4r1_2t, y4_4r0_2t, y4_4rf1_2t, y4_4rf2_2t],dim=1)
            # xd8s = torch.cat([y4_4r4_2t, y4_4r3_2t, y4_4r2_2t, y4_4r1_2t, y4_4r0_2t, y4_4rf1_2t, y4_4rf2_2t, y4_4rf3_2t, y4_4rf4_2t], dim=1)
            # print('xd8', xd8.shape)

            if i == 0:
                xd1 = xd1s.unsqueeze(dim=0)
                xd2 = xd2s.unsqueeze(dim=0)
                xd3 = xd3s.unsqueeze(dim=0)
                xd4 = xd4s.unsqueeze(dim=0)
                xd5 = xd5s.unsqueeze(dim=0)
                xd6 = xd6s.unsqueeze(dim=0)
                xd7 = xd7s.unsqueeze(dim=0)
                xd8 = xd8s.unsqueeze(dim=0)
            else:
                xd1 = torch.cat([xd1, xd1s.unsqueeze(dim=0)], dim=0)
                xd2 = torch.cat([xd2, xd2s.unsqueeze(dim=0)], dim=0)
                xd3 = torch.cat([xd3, xd3s.unsqueeze(dim=0)], dim=0)
                xd4 = torch.cat([xd4, xd4s.unsqueeze(dim=0)], dim=0)
                xd5 = torch.cat([xd5, xd5s.unsqueeze(dim=0)], dim=0)
                xd6 = torch.cat([xd6, xd6s.unsqueeze(dim=0)], dim=0)
                xd7 = torch.cat([xd7, xd7s.unsqueeze(dim=0)], dim=0)
                xd8 = torch.cat([xd8, xd8s.unsqueeze(dim=0)], dim=0)
        i = i + 1
        x1 = x1_1
        x1r = x1.reshape(x1.shape[0], x1.shape[1], -1)
        #
        # # x2 = Variable(x1r.cpu())
        # # x2 = Variable(x1r).cpu()
        x2 = x1r.cpu()
        x2rn = np.flip(x2.numpy(), axis=2).copy()
        x2rt = torch.from_numpy(x2rn)
        x2r = x2rt.cuda()
        #
        x3 = torch.transpose(x1, 2, 3)
        x3r = x3.reshape(x3.shape[0], x3.shape[1], -1)
        #
        # # x4 = Variable(x3r.cpu())
        # # x4 = Variable(x3r).cpu()
        x4 = x3r.cpu()
        x4rn = np.flip(x4.numpy(), axis=2).copy()
        x4rt = torch.from_numpy(x4rn)
        x4r = x4rt.cuda()
        #
        x5 = torch.rot90(x1,1,(2,3))
        x5r = x5.reshape(x5.shape[0], x5.shape[1], -1)
        #
        # # x6 = Variable(x5r.cpu())
        # # x6 = Variable(x5r).cpu()
        x6 = x5r.cpu()
        x6rn = np.flip(x6.numpy(), axis=2).copy()
        x6rt = torch.from_numpy(x6rn)
        x6r = x6rt.cuda()
        #
        x7 = torch.transpose(x5,2,3)
        x7r = x7.reshape(x7.shape[0], x7.shape[1], -1)
        #
        # # x8 = Variable(x7r.cpu())
        # # x8 = Variable(x7r).cpu()
        x8 = x7r.cpu()
        x8rn = np.flip(x8.numpy(), axis=2).copy()
        x8rt = torch.from_numpy(x8rn)
        x8r = x8rt.cuda()

        xd8 = xd8.permute(2, 0, 1)
        xd7 = xd7.permute(2, 0, 1)
        xd6 = xd6.permute(2, 0, 1)
        xd5 = xd5.permute(2, 0, 1)
        xd4 = xd4.permute(2, 0, 1)
        xd3 = xd3.permute(2, 0, 1)
        xd2 = xd2.permute(2, 0, 1)
        xd1 = xd1.permute(2, 0, 1)

        x8r = x8r.permute(2, 0, 1)
        x7r = x7r.permute(2, 0, 1)
        x6r = x6r.permute(2, 0, 1)
        x5r = x5r.permute(2, 0, 1)
        x4r = x4r.permute(2, 0, 1)
        x3r = x3r.permute(2, 0, 1)
        x2r = x2r.permute(2, 0, 1)
        x1r = x1r.permute(2, 0, 1)

        # print('into GRU', x3r.shape)
        x1r_r = self.gru_2_1(x1r)[0]
        x1r_r_last = x1r_r[-1]
        x1r_r_last = x1r_r_last.permute(0, 1).contiguous()
        x1r_r_last_b1 = x1r_r_last[1,:]
        x1r_r_last_b1 = x1r_r_last_b1.unsqueeze(0)
        # plt.subplot(4,4,1).set_title('direction-1')
        # plt.plot(x1r_r_last_b1.cpu().detach().numpy())
        # x1r_r_last = F.tanh(x1r_r_last)

        xd1r_r = self.gru_2_9(xd1)[0]
        xd1r_r_last = xd1r_r[-1]
        xd1r_r_last = xd1r_r_last.permute(0, 1).contiguous()
        # print(x1r_r_last_b1.shape)
        xd1r_r_last_b1 = xd1r_r_last[1,:]
        xd1r_r_last_b1 = xd1r_r_last_b1.unsqueeze(0)
        # plt.subplot(4, 4, 9).set_title('diagnol-1')
        # plt.plot(xd1r_r_last_b1.cpu().detach().numpy())
        # xd1r_r = F.tanh(xd1r_r)

        x2r_r = self.gru_2_2(x2r)[0]
        x2r_r_last = x2r_r[-1]
        x2r_r_last = x2r_r_last.permute(0, 1).contiguous()
        x2r_r_last_b1 = x2r_r_last[1,:]
        x2r_r_last_b1 = x2r_r_last_b1.unsqueeze(0)
        # plt.subplot(4, 4, 2).set_title('direction-2')
        # plt.plot(x2r_r_last_b1.cpu().detach().numpy())
        # x2r_r = F.tanh(x2r_r)

        xd2r_r = self.gru_2_10(xd2)[0]
        xd2r_r_last = xd2r_r[-1]
        xd2r_r_last = xd2r_r_last.permute(0, 1).contiguous()
        xd2r_r_last_b1 = xd2r_r_last[1,:]
        xd2r_r_last_b1 = xd2r_r_last_b1.unsqueeze(0)
        # plt.subplot(4, 4, 10).set_title('diagnol-2')
        # plt.plot(xd2r_r_last_b1.cpu().detach().numpy())
        # xd2r_r = F.tanh(xd2r_r)

        x3r_r = self.gru_2_3(x3r)[0]
        x3r_r_last = x3r_r[-1]
        x3r_r_last = x3r_r_last.permute(0, 1).contiguous()
        x3r_r_last_b1 = x3r_r_last[1,:]
        x3r_r_last_b1 = x3r_r_last_b1.unsqueeze(0)
        # plt.subplot(4, 4, 3).set_title('direction-3')
        # plt.plot(x3r_r_last_b1.cpu().detach().numpy())
        # x3r_r = F.tanh(x3r_r)

        xd3r_r = self.gru_2_11(xd3)[0]
        xd3r_r_last = xd3r_r[-1]
        xd3r_r_last = xd3r_r_last.permute(0, 1).contiguous()
        xd3r_r_last_b1 = xd3r_r_last[1,:]
        xd3r_r_last_b1 =xd3r_r_last_b1.unsqueeze(0)
        # plt.subplot(4, 4, 11).set_title('diagnol-3')
        # plt.plot(xd3r_r_last_b1.cpu().detach().numpy())
        # xd3r_r = F.tanh(xd3r_r)

        x4r_r = self.gru_2_4(x4r)[0]
        x4r_r_last = x4r_r[-1]
        x4r_r_last = x4r_r_last.permute(0, 1).contiguous()
        x4r_r_last_b1 = x4r_r_last[1,:]
        x4r_r_last_b1 = x4r_r_last_b1.unsqueeze(0)
        # plt.subplot(4, 4, 4).set_title('direction-4')
        # plt.plot(x4r_r_last_b1.cpu().detach().numpy())
        # x4r_r = F.tanh(x4r_r)

        xd4r_r = self.gru_2_12(xd4)[0]
        xd4r_r_last = xd4r_r[-1]
        xd4r_r_last = xd4r_r_last.permute(0, 1).contiguous()
        xd4r_r_last_b1 = xd4r_r_last[1,:]
        xd4r_r_last_b1 = xd4r_r_last_b1.unsqueeze(0)
        # plt.subplot(4, 4, 12).set_title('diagnol-4')
        # plt.plot(xd4r_r_last_b1.cpu().detach().numpy())
        # xd4r_r = F.tanh(xd4r_r)

        x5r_r = self.gru_2_5(x5r)[0]
        x5r_r_last = x5r_r[-1]
        x5r_r_last = x5r_r_last.permute(0, 1).contiguous()
        x5r_r_last_b1 = x5r_r_last[1,:]
        x5r_r_last_b1 = x5r_r_last_b1.unsqueeze(0)
        # plt.subplot(4, 4, 5).set_title('direction-5')
        # plt.plot(x5r_r_last_b1.cpu().detach().numpy())
        # x5r_r = F.tanh(x5r_r)

        xd5r_r = self.gru_2_13(xd5)[0]
        xd5r_r_last = xd5r_r[-1]
        xd5r_r_last = xd5r_r_last.permute(0, 1).contiguous()
        xd5r_r_last_b1 = xd5r_r_last[1,:]
        xd5r_r_last_b1 = xd5r_r_last_b1.unsqueeze(0)
        # plt.subplot(4, 4, 13).set_title('diagnol-5')
        # plt.plot(xd5r_r_last_b1.cpu().detach().numpy())
        # xd5r_r = F.tanh(xd5r_r)

        x6r_r = self.gru_2_6(x6r)[0]
        x6r_r_last = x6r_r[-1]
        x6r_r_last = x6r_r_last.permute(0, 1).contiguous()
        x6r_r_last_b1 = x6r_r_last[1,:]
        x6r_r_last_b1 = x6r_r_last_b1.unsqueeze(0)
        # plt.subplot(4, 4, 6).set_title('direction-6')
        # plt.plot(x6r_r_last_b1.cpu().detach().numpy())
        # x6r_r = F.tanh(x6r_r)

        xd6r_r = self.gru_2_14(xd6)[0]
        xd6r_r_last = xd6r_r[-1]
        xd6r_r_last = xd6r_r_last.permute(0, 1).contiguous()
        xd6r_r_last_b1 = xd6r_r_last[1,:]
        xd6r_r_last_b1 = xd6r_r_last_b1.unsqueeze(0)
        # plt.subplot(4, 4, 14).set_title('diagnol-6')
        # plt.plot(xd6r_r_last_b1.cpu().detach().numpy())
        # xd6r_r = F.tanh(xd6r_r)

        x7r_r = self.gru_2_7(x7r)[0]
        x7r_r_last = x7r_r[-1]
        x7r_r_last = x7r_r_last.permute(0, 1).contiguous()
        x7r_r_last_b1 = x7r_r_last[1,:]
        x7r_r_last_b1 = x7r_r_last_b1.unsqueeze(0)
        # plt.subplot(4, 4, 7).set_title('direction-7')
        # plt.plot(x7r_r_last_b1.cpu().detach().numpy())
        # x7r_r = F.tanh(x7r_r)

        xd7r_r = self.gru_2_15(xd7)[0]
        xd7r_r_last = xd7r_r[-1]
        xd7r_r_last = xd7r_r_last.permute(0, 1).contiguous()
        xd7r_r_last_b1 = xd7r_r_last[1,:]
        xd7r_r_last_b1 = xd7r_r_last_b1.unsqueeze(0)
        # plt.subplot(4, 4, 15).set_title('diagnol-7')
        # plt.plot(xd7r_r_last_b1.cpu().detach().numpy())
        # xd7r_r = F.tanh(xd7r_r)
        # print('x7dr_r', xd7r_r.shape)

        x8r_r = self.gru_2_8(x8r)[0]
        x8r_r_last = x8r_r[-1]
        x8r_r_last = x8r_r_last.permute(0, 1).contiguous()
        x8r_r_last_b1 = x8r_r_last[1,:]
        x8r_r_last_b1 = x8r_r_last_b1.unsqueeze(0)
        # plt.subplot(4, 4, 8).set_title('direction-8')
        # plt.plot(x8r_r_last_b1.cpu().detach().numpy())
        # x8r_r = F.tanh(x8r_r)
        # print('x8r_r', x8r_r.shape)

        xd8r_r = self.gru_2_16(xd8)[0]
        xd8r_r_last = xd8r_r[-1]
        xd8r_r_last = xd8r_r_last.permute(0, 1).contiguous()
        xd8r_r_last_b1 = xd8r_r_last[1,:]
        xd8r_r_last_b1 = xd8r_r_last_b1.unsqueeze(0)

        # plt.subplot(4, 4, 16).set_title('diagnol-8')
        # plt.plot(xd8r_r_last_b1.cpu().detach().numpy())
        # x8r_r = self.gru2(x8r + x7r_r + x6r)[0]
        # xd8r_r = F.tanh(xd8r_r)
        # x8r_r = F.relu(x8r_r)
        # print('xd8r_r',xd8r_r.shape)
        M = torch.cat([x1r_r_last_b1,xd1r_r_last_b1, x2r_r_last_b1,xd2r_r_last_b1,x3r_r_last_b1,xd3r_r_last_b1,x4r_r_last_b1,xd4r_r_last_b1,
                       x5r_r_last_b1,xd5r_r_last_b1,x6r_r_last_b1,xd6r_r_last_b1,x7r_r_last_b1,xd7r_r_last_b1,x8r_r_last_b1,xd8r_r_last_b1], dim=0)
        print('M.shape', M.shape)
        pca = PCA(n_components=2)
        M = M.cpu().detach().numpy()
        reduced = pca.fit_transform(M)
        print(reduced.shape)
        # t = reduced
        t = reduced.transpose()
        print(t.shape)
        cValue = ['b', 'b','c','c','g', 'g','k', 'k','m', 'm','r', 'r','#FF8C00', '#FF8C00','y', 'y']  # blue第一,cyan第二,绿色第三,black_4,紫色_5,红色6,橘色7,黄色8
        plt.scatter(t[0], t[1], s=100, cmap=True, c=cValue, marker='^')
        # plt.legend()
        plt.show()


        
        xd8r_r = xd8r_r.permute(1, 2, 0).contiguous()
        print(xd8r_r.shape)
        # x7r = x7r.permute(1, 2, 0).contiguous()
        # x6r = x6r.permute(1, 2, 0).contiguous()
        # x5r = x5r.permute(1, 2, 0).contiguous()
        # x4r = x4r.permute(1, 2, 0).contiguous()
        # x3r = x3r.permute(1, 2, 0).contiguous()
        # x2r = x2r.permute(1, 2, 0).contiguous()
        # x1r = x1r.permute(1, 2, 0).contiguous()

        xd8r_r = xd8r_r.view(xd8r_r.size(0), -1)
        # x7r = x7r.view(x7r.size(0), -1)
        # x6r = x6r.view(x6r.size(0), -1)
        # x5r = x5r.view(x5r.size(0), -1)
        # x4r = x4r.view(x4r.size(0), -1)
        # x3r = x3r.view(x3r.size(0), -1)
        # x2r = x2r.view(x2r.size(0), -1)
        # x1r = x1r.view(x1r.size(0), -1)
        # print('view_size', x1r.shape)
        x = xd8r_r

        # x = x8r + x7r + x6r + x5r + x4r + x3r + x2r + x1r
        # w1 = x1r / x
        # w2 = x2r / x
        # w3 = x3r / x
        # w4 = x4r / x
        # w5 = x5r / x
        # w6 = x6r / x
        # w7 = x7r / x
        # w8 = x8r / x
        # x = 0.35 * x1 + 0.35 * x2 + 0.15 * x3 + 0.15 * x4
        # x = w1*x1r + w2*x2r + w3*x3r + w4*x4r + w5*x5r + w6*x6r + w7*x7r + w8*x8r
        # x = w1 * x1r + w2 * x2r + w3 * x3r + w4 * x4r + w5 * x5r + w6 * x6r + w7 * x7r + w8 * x8r
        # x = 0.125 * x1r + 0.125*x2r + 0.125*x3r + 0.125*x4r + 0.125*x5r + 0.125*x6r + 0.125*x7r + 0.125*x8r
        # print('into gru_bn', x.shape)
        # x = self.gru_bn(x)
        x = self.gru_bn_2(x)
        # x = self.gru_bn3(x)
        x = self.relu(x)
        # x = self.tanh(x)
        # x = self.elu(x)
        # x = self.prelu(x)
        # print('into fc', x.shape)
        x = self.dropout(x)
        # x = self.fc(x)
        x = self.fc_2(x)
        # x = self.fc3(x)
        plt.show()
        return x

class zhouOneDRNN(nn.Module):
    """
    one direction rnn with spatial consideration which has a patch size
    """
    @staticmethod
    def weight_init(m):
        # All weight matrices in our RNN and bias vectors are initialized with a uniform distribution, and the values of these weight matrices and bias vectors are initialized in the range [−0.1,0.1]
        if isinstance(m, (nn.Linear, nn.GRU, nn.LSTM)):
            init.kaiming_uniform_(m.weight.data)
            init.kaiming_uniform_(m.bias.data)

    def __init__(self, input_channels, n_classes, patch_size=5):
        # The proposed network model uses a single recurrent layer that adopts our modified GRUs of size 64 with sigmoid gate activation and PRetanh activation functions for hidden representations
        super(zhouOneDRNN, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.lstm_seq = nn.LSTM(input_channels,patch_size**2,1)
        self.gru_2 = nn.GRU(input_channels,patch_size**2,1,bidirectional=False)
        self.gru_2_1 = nn.LSTM(input_channels, patch_size ** 2, 1, bidirectional=False)
        self.gru_2_2 = nn.LSTM(input_channels, patch_size ** 2, 1, bidirectional=False)
        self.gru_2_3 = nn.GRU(input_channels, patch_size ** 2, 1, bidirectional=False)
        self.gru_2_4 = nn.GRU(input_channels, patch_size ** 2, 1, bidirectional=False)
        self.gru = nn.GRU(patch_size**2, patch_size**2, 1, bidirectional=False, batch_first=False)  # TODO: try to change this ?
        self.gru_bn = nn.BatchNorm1d((patch_size**2) * input_channels)
        self.gru_bn_2 = nn.BatchNorm1d((patch_size**2) * (patch_size**2))
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.fc = nn.Linear((patch_size**2) * input_channels, n_classes)
        self.fc_2 = nn.Linear((patch_size**2) * (patch_size**2), n_classes)
        self.rec_2 = nn.Linear((patch_size ** 2) * (patch_size ** 2), input_channels)
        self.pooling = nn.MaxPool2d(3,3)
        self.dropout = nn.Dropout(0.5)
        self.aux_loss_weight = 1

    def visualize_difference_image(self, differences, sequence_length):
        # differences is expected to be a tensor of size (batch, length)
        for i in range(differences.size(0)):  # for each sequence in the batch
            # reshape differences to 2D
            diff_image = differences[i].view(int(np.sqrt(sequence_length)), -1).cpu().numpy()

            plt.figure()
            plt.imshow(diff_image, cmap='hot', interpolation='nearest')
            plt.title(f'Sequence {i + 1} - Pixel Difference')
            plt.colorbar(label='Pixel Difference')
            plt.show()


    def forward(self, x):  # 初始是第三方向
        # print("x",x.shape)
        x = x.squeeze(1)
        # plt.imshow(x[0, [35, 24, 5], :, :].permute(1, 2, 0).cpu().numpy())
        # plt.show()
        # print("x", x.shape)
        # 生成第一方向
        x1 = torch.transpose(x, 2, 3)
        x1r = x.reshape(x1.shape[0],x1.shape[1],-1)
        # print('0', x1r.shape)
        #生成第二方向
        x2 = x1r.cpu()
        x2rn = np.flip(x2.numpy(), axis=2).copy()
        x2rt = torch.from_numpy(x2rn)
        x2r = x2rt.cuda()
        # print('2',x.shape)
        #生成第三方向
        x3 = torch.transpose(x1, 2, 3)
        x3r = x3.reshape(x3.shape[0], x3.shape[1], -1)
        # # #生成第四方向 从第三方向来
        x4 = x3r.cpu()
        x4rn = np.flip(x4.numpy(), axis=2).copy()
        x4rt = torch.from_numpy(x4rn)
        x4r = x4rt.cuda()
        #生成第五方向
        x5 = torch.rot90(x1, 1, (2, 3))
        x5r = x5.reshape(x5.shape[0], x5.shape[1], -1)
        # #生成第六方向
        x6 = x5r.cpu()
        x6rn = np.flip(x6.numpy(), axis=2).copy()
        x6rt = torch.from_numpy(x6rn)
        x6r = x6rt.cuda()
        # #生成第七方向
        x7 = torch.transpose(x5, 2, 3)
        x7r = x7.reshape(x7.shape[0], x7.shape[1], -1)
        # #生成第八方向
        x8 = x7r.cpu()
        x8rn = np.flip(x8.numpy(), axis=2).copy()
        x8rt = torch.from_numpy(x8rn)
        x8r = x8rt.cuda()
        #下面改变输入值，确定使用哪个方向
        x1r = x1r.permute(2, 0, 1)
        x2r = x2r.permute(2, 0, 1)
        x3r = x3r.permute(2, 0, 1)
        x4r = x4r.permute(2, 0, 1)
        x5r = x5r.permute(2, 0, 1)
        x6r = x6r.permute(2, 0, 1)
        x7r = x7r.permute(2, 0, 1)
        x8r = x8r.permute(2, 0, 1) # s  b  c
        # print('x1r', x1r.shape)
        # plt.subplot(1, 2, 1)
        # plt.plot(x1r[-1, :, :].cpu().detach().numpy())
        # plt.grid(linewidth=0.5, color='black')
        # plt.title('Real situation in one patch', fontdict={'size': 40})
        # plt.xlabel('Band Numbers', fontdict={'size': 40}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 40})
        # plt.xticks(fontsize=35)
        # plt.yticks(fontsize=35)
        # bwith = 2  # 边框宽度设置为2
        # TK = plt.gca()  # 获取边框
        # TK.spines['bottom'].set_linewidth(bwith)  # 图框下边
        # TK.spines['left'].set_linewidth(bwith)  # 图框左边
        # TK.spines['top'].set_linewidth(bwith)  # 图框上边
        # TK.spines['right'].set_linewidth(bwith)  # 图框右边
        # print(x.shape)

        # multi_1 = multiscan(rearrange(x, 'b c w h -> b h w c'), 1)
        # multi_2 = multiscan(rearrange(x, 'b c w h -> b h w c'), 2)
        # multi_3 = multiscan(rearrange(x, 'b c w h -> b h w c'), 3)
        # multi_4 = multiscan(rearrange(x, 'b c w h -> b h w c'), 4)
        # print('multi', multi_1.shape) #multi torch.Size([100, 103, 25])

        # diagonal_1 = diagonal_flatten_image(x1.cpu())
        # print('diagonal size', diagonal_1.shape) #100, 103, 25
        # perimeter_1 = perimeter_scan_batch(x.cpu())
        # perimeter_2 = perimeter_scan_batch(x1.cpu())
        # print('perimeter', perimeter_1.shape)


        # print('x',x1r.shape)
        # entr1 = torch.special.entr(x1r)
        # entr2 = torch.special.entr(x2r)
        # entr3 = torch.special.entr(x3r)
        # entr4 = torch.special.entr(x4r)
        #
        # entr_multi_1 = torch.special.entr(rearrange(multi_1, 'b c s -> s b c'))
        # entr_multi_2 = torch.special.entr(rearrange(multi_2, 'b c s -> s b c'))
        # entr_multi_3 = torch.special.entr(rearrange(multi_3, 'b c s -> s b c'))
        # entr_multi_4 = torch.special.entr(rearrange(multi_4, 'b c s -> s b c'))

        #
        # plt.subplot(241)
        # sns.heatmap(entr1[:,0,:].cpu().numpy())
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.xlabel('Entropy alongside spectral', fontsize=20)
        # plt.ylabel('Entropy alongside spatial', fontsize=20)
        # plt.subplot(242)
        # sns.heatmap(entr2[:, 0, :].cpu().numpy())
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.xlabel('Entropy alongside spectral', fontsize=20)
        # plt.ylabel('Entropy alongside spatial', fontsize=20)
        # plt.subplot(243)
        # sns.heatmap(entr3[:, 0, :].cpu().numpy())
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.xlabel('Entropy alongside spectral', fontsize=20)
        # plt.ylabel('Entropy alongside spatial', fontsize=20)
        # plt.subplot(244)
        # sns.heatmap(entr4[:, 0, :].cpu().numpy())
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.xlabel('Entropy alongside spectral', fontsize=20)
        # plt.ylabel('Entropy alongside spatial', fontsize=20)
        # plt.subplot(245)
        # sns.heatmap(entr_multi_1[:, 0, :].cpu().numpy())
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.xlabel('Entropy alongside spectral', fontsize=20)
        # plt.ylabel('Entropy alongside spatial', fontsize=20)
        # plt.subplot(246)
        # sns.heatmap(entr_multi_2[:, 0, :].cpu().numpy())
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.xlabel('Entropy alongside spectral', fontsize=20)
        # plt.ylabel('Entropy alongside spatial', fontsize=20)
        # plt.subplot(247)
        # sns.heatmap(entr_multi_3[:, 0, :].cpu().numpy())
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.xlabel('Entropy alongside spectral', fontsize=20)
        # plt.ylabel('Entropy alongside spatial', fontsize=20)
        # plt.subplot(248)
        # sns.heatmap(entr_multi_4[:, 0, :].cpu().numpy())
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.xlabel('Entropy alongside spectral', fontsize=20)
        # plt.ylabel('Entropy alongside spatial', fontsize=20)
        # plt.show()
        #
        # plt.subplot(241)
        # pairwise_distances_1 = torch.cdist(rearrange(x1r, 'n b d -> b n d'), rearrange(x1r, 'n b d -> b n d'), p=2)
        # sns.heatmap(pairwise_distances_1[0, :, :].cpu().numpy())
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.xlabel('Pixel index', fontsize=20)
        # plt.ylabel('Pixel index', fontsize=20)
        # plt.subplot(242)
        # pairwise_distances_2 = torch.cdist(rearrange(x2r, 'n b d -> b n d'), rearrange(x2r, 'n b d -> b n d'), p=2)
        # sns.heatmap(pairwise_distances_2[0, :, :].cpu().numpy())
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.xlabel('Pixel index', fontsize=20)
        # plt.ylabel('Pixel index', fontsize=20)
        # plt.subplot(243)
        # pairwise_distances_3 = torch.cdist(rearrange(x3r, 'n b d -> b n d'), rearrange(x3r, 'n b d -> b n d'), p=2)
        # sns.heatmap(pairwise_distances_3[0, :, :].cpu().numpy())
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.xlabel('Pixel index', fontsize=20)
        # plt.ylabel('Pixel index', fontsize=20)
        # plt.subplot(244)
        # pairwise_distances_4 = torch.cdist(rearrange(x4r, 'n b d -> b n d'), rearrange(x4r, 'n b d -> b n d'), p=2)
        # sns.heatmap(pairwise_distances_4[0, :, :].cpu().numpy())
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.xlabel('Pixel index', fontsize=20)
        # plt.ylabel('Pixel index', fontsize=20)
        # plt.subplot(245)
        # pairwise_distances_5 = torch.cdist(rearrange(multi_1, 'b c s -> b s c'), rearrange(multi_1, 'b c s -> b s c'), p=2)
        # sns.heatmap(pairwise_distances_5[0, :, :].cpu().numpy())
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.xlabel('Pixel index', fontsize=20)
        # plt.ylabel('Pixel index', fontsize=20)
        # plt.subplot(246)
        # pairwise_distances_6 = torch.cdist(rearrange(multi_2, 'b c s -> b s c'), rearrange(multi_2, 'b c s -> b s c'), p=2)
        # sns.heatmap(pairwise_distances_6[0, :, :].cpu().numpy())
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.xlabel('Pixel index', fontsize=20)
        # plt.ylabel('Pixel index', fontsize=20)
        # plt.subplot(247)
        # pairwise_distances_7 = torch.cdist(rearrange(multi_3, 'b c s -> b s c'), rearrange(multi_3, 'b c s -> b s c'), p=2)
        # sns.heatmap(pairwise_distances_7[0, :, :].cpu().numpy())
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.xlabel('Pixel index', fontsize=20)
        # plt.ylabel('Pixel index', fontsize=20)
        # plt.subplot(248)
        # pairwise_distances_8 = torch.cdist(rearrange(multi_4, 'b c s -> b s c'), rearrange(multi_4, 'b c s -> b s c'), p=2)
        # sns.heatmap(pairwise_distances_8[0, :, :].cpu().numpy())
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.xlabel('Pixel index', fontsize=20)
        # plt.ylabel('Pixel index', fontsize=20)
        # plt.show()

        #导入GRU
        # x = self.gru(x1r+x2r)[0]
        x1 = self.gru_2_1(x1r)
        x2 = self.gru_2_2(x2r)
        x3 = self.gru_2_3(x3r)
        x4 = self.gru_2_4(x4r)
        # x1_diag = self.gru_2_1(rearrange(diagonal_1.cuda(), 'b c s ->s b c'))
        # x1_uturn= self.gru_2_1(rearrange(multi_1.cuda(), 'b c s ->s b c'))
        # x1_perimeter = self.gru_2_1(rearrange(perimeter_1.cuda().float(), 'b c s ->s b c'))
        # x2_perimeter = self.gru_2_2(rearrange(perimeter_2.cuda().float(), 'b c s ->s b c'))
        # print('x_strategy_1_laststep', x_strategy_1_laststep.shape)
        x = x1
        x_2 = x2
        x_3 = x3
        x_4 = x4
        # c = int(x.size(0)-1)
        # x_strategy_1_laststep = x[-1]

        # np.save('x_strategy_1_laststep', x_strategy_1_laststep.cpu().detach().numpy(), allow_pickle=True)
        # x2r = self.gru_2(x2r)[0]
        # x = x1r + x2r
        # print('out',out.shape) #(103,16,64)
        x = x.permute(1, 2, 0).contiguous()
        x_2 = x_2.permute(1, 2, 0).contiguous()
        x_3 = x_3.permute(1, 2, 0).contiguous()
        x_4 = x_4.permute(1, 2, 0).contiguous()
        # print(x.shape) #(16,64,103)
        x = x.view(x.size(0),-1)
        x_2 = x_2.view(x_2.size(0), -1)
        x_3 = x_3.view(x_3.size(0), -1)
        x_4 = x_4.view(x_4.size(0), -1)
        # print('5',x.shape) #(16,6592)
        # plt.subplot(3,2,2)
        # plt.plot(x[-1,:].cpu().detach().numpy())
        # x = self.gru_bn(x)
        x = self.gru_bn_2(x)
        x_2 = self.gru_bn_2(x_2)
        x_3 = self.gru_bn_2(x_3)
        x_4 = self.gru_bn_2(x_4)
        # plt.subplot(3, 2, 3)
        # plt.plot(x[-1, :].cpu().detach().numpy())
        x = self.relu(x)
        x_2 = self.relu(x_2)
        x_3 = self.relu(x_3)
        x_4 = self.relu(x_4)
        # plt.subplot(3, 2, 4)
        # plt.plot(x[-1, :].cpu().detach().numpy())
        x = self.dropout(x)
        x_2 = self.dropout(x_2)
        x_3 = self.dropout(x_3)
        x_4 = self.dropout(x_4)
        # plt.subplot(3, 2, 5)
        # plt.plot(x[-1, :].cpu().detach().numpy())
        # x = self.fc(x)
        x_class = self.fc_2(x)
        x_2class = self.fc_2(x_2)
        x_3class = self.fc_2(x_3)
        x_4class = self.fc_2(x_4)
        x_rec = self.rec_2(x + x_2 + x_3 + x_4)
        # x = self.softmax(x)
        # plt.subplot(1, 2, 2)
        # plt.plot(x[-1, :].cpu().detach().numpy(),linewidth=2.5)
        # plt.grid(linewidth=0.5, color='black')
        # plt.title('Real situation in one patch', fontdict={'size': 40})
        # plt.xlabel('Band Numbers', fontdict={'size': 40}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 40})
        # plt.xticks(fontsize=35)
        # plt.yticks(fontsize=35)
        # bwith = 2  # 边框宽度设置为2
        # TK = plt.gca()  # 获取边框
        # TK.spines['bottom'].set_linewidth(bwith)  # 图框下边
        # TK.spines['left'].set_linewidth(bwith)  # 图框左边
        # TK.spines['top'].set_linewidth(bwith)  # 图框上边
        # TK.spines['right'].set_linewidth(bwith)  # 图框右边
        # plt.show()
        # print('after fc',x.shape)
        # x = x + x_2
        return x_class, x_rec

class zhouFourDRNN(nn.Module):

    @staticmethod
    def weight_init(m):
        # All weight matrices in our RNN and bias vectors are initialized with a uniform distribution, and the values of these weight matrices and bias vectors are initialized in the range [−0.1,0.1]
        if isinstance(m, (nn.Linear, nn.GRU, nn.Conv3d)):
            init.kaiming_uniform_(m.weight.data)
            init.kaiming_uniform_(m.bias.data)

    def __init__(self, input_channels, n_classes, patch_size=5):
        # The proposed network model uses a single recurrent layer that adopts our modified GRUs of size 64 with sigmoid gate activation and PRetanh activation functions for hidden representations
        super(zhouFourDRNN, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.gru = nn.GRU(patch_size**2 , patch_size**2, 1, bidirectional=False, batch_first=False)  # TODO: try to change this ?
        self.gru_2 = nn.LSTM(input_channels, input_channels, 1, bidirectional=False)
        self.gru_3 = nn.LSTM(input_channels, patch_size**2, 1, bidirectional=False)
        self.gru_bn = nn.BatchNorm1d(patch_size**2 * input_channels)
        self.gru_bn_2 = nn.BatchNorm1d((patch_size ** 2) * input_channels)
        self.gru_bn_3 = nn.BatchNorm1d((patch_size ** 2) * (patch_size**2))
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(patch_size**2 * input_channels , n_classes)
        self.fc_2 = nn.Linear((patch_size ** 2) * input_channels, n_classes)
        self.fc_3 = nn.Linear((patch_size ** 2) * (patch_size**2), n_classes)
        self.rec_3 = nn.Linear((patch_size ** 2) * (patch_size ** 2), input_channels)
        self.softmax = nn.Softmax()
        self.aux_loss_weight = 1

    def forward(self, x): #初始是第1方向
        x = x.squeeze(1)
        # print('0', x.shape)
        x1 = x
        x1r = x1.reshape(x1.shape[0], x1.shape[1], -1)
        # plt.plot(x1r[0,:,:].cpu().detach().numpy())
        # plt.plot(x1r[0,:,12].cpu().detach().numpy(), linewidth=5)
        # plt.show()

        # x2 = Variable(x1r.cpu())
        # x2 = Variable(x1r).cpu()
        x2 = x1r.cpu()
        x2rn = np.flip(x2.numpy(), axis=2).copy()
        x2rt = torch.from_numpy(x2rn)
        x2r = x2rt.cuda()

        x3 = torch.transpose(x1, 2, 3)
        x3r = x3.reshape(x3.shape[0], x3.shape[1], -1)

        # x4 = Variable(x3r.cpu())
        # x4 = Variable(x3r).cpu()
        x4 = x3r.cpu()
        x4rn = np.flip(x4.numpy(), axis=2).copy()
        x4rt = torch.from_numpy(x4rn)
        x4r = x4rt.cuda()

        x5 = torch.rot90(x1, 1, (2, 3))
        x5r = x5.reshape(x5.shape[0], x5.shape[1], -1)

        # x6 = Variable(x5r.cpu())
        # x6 = Variable(x5r).cpu()
        x6 = x5r.cpu()
        x6rn = np.flip(x6.numpy(), axis=2).copy()
        x6rt = torch.from_numpy(x6rn)
        x6r = x6rt.cuda()

        x7 = torch.transpose(x5, 2, 3)
        x7r = x7.reshape(x7.shape[0], x7.shape[1], -1)

        # x8 = Variable(x7r.cpu())
        # x8 = Variable(x7r).cpu()
        x8 = x7r.cpu()
        x8rn = np.flip(x8.numpy(), axis=2).copy()
        x8rt = torch.from_numpy(x8rn)
        x8r = x8rt.cuda()
        # print('x8r',x8r.shape) #(16,103,25)
        # plt.plot(x1r[0,:,:].cpu().detach().numpy())

        x8r = x8r.permute(2, 0, 1) #(25,16,103)
        x7r = x7r.permute(2, 0, 1)
        x6r = x6r.permute(2, 0, 1)
        x5r = x5r.permute(2, 0, 1)
        x4r = x4r.permute(2, 0, 1)
        x3r = x3r.permute(2, 0, 1)
        x2r = x2r.permute(2, 0, 1)
        x1r = x1r.permute(2, 0, 1)

        plt.figure(figsize=(7, 6))
        plt.plot(x1r[1, 0, :].cpu().numpy(), lw=3.5, color='Red')
        plt.title('Spectral signal of one pixel', fontsize=20)
        plt.xticks(np.arange(1, 201, 19).astype(int), fontsize=20)  # Set x-axis ticks from 1 to 200
        plt.ylabel('Spectral Value', fontsize=20)
        plt.yticks(fontsize=20)  # Increase y-axis tick label font size to 12
        plt.xlabel('Band Number', fontsize=20)
        # Adding the bounding box
        ax = plt.gca()
        box = patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=None, edgecolor='black', linewidth=5)
        ax.add_patch(box)
        plt.show()

        # print('x8r', x8r.shape) #(25,16,103)
        # plt.subplot(1,5,1)
        # plt.subplot(1, 5, 2)
        # plt.plot(x2r[:, 0, :].cpu().detach().numpy())
        x1r_r = self.gru_2(x1r)
        x1r_r = self.relu(x1r_r)
        # print('x1r',x1r.shape) #(25,16,103)
        # plt.subplot(1,5,1)
        # plt.plot(x1r[:, 0, :].cpu().detach().numpy())
        # plt.subplot(1,5,4)
        # plt.plot((x1r+x2r)[:, 0, :].cpu().detach().numpy())
        x2r_r = self.gru_2(x2r + x1r_r) #把x1r经过RNN的值，作为x2r的输入
        x2r_r = self.relu(x2r_r)
        # plt.subplot(1,5,2)
        # plt.plot(x2r[:, 0, :].cpu().detach().numpy())
        x3r_r = self.gru_2(x3r + x2r_r)
        x3r_r = self.relu(x3r_r)
        # plt.subplot(1, 5, 3)
        # plt.plot(x3r[:, 0, :].cpu().detach().numpy())
        x4r_r = self.gru_2(x4r + x3r_r)
        x4r_r = self.relu(x4r_r)
        # plt.subplot(1, 5, 4)
        # plt.plot(x4r[:, 0, :].cpu().detach().numpy())
        x5r_r = self.gru_2(x5r + x4r_r)
        x5r_r = self.relu(x5r_r)
        #
        x6r_r = self.gru_2(x6r + x5r_r)
        x6r_r = self.relu(x6r_r)
        #
        x7r_r = self.gru_2(x7r + x6r_r)
        x7r_r = self.relu(x7r_r)
        #
        x8r_r = self.gru_2(x8r + x7r_r)
        # x8r_r = self.relu(x8r_r)
        # x8r = self.gru(x8r+x7r)[0]
        x = x1r_r
        # print(x.shape)
        x = self.gru_3(x)
        # x = self.gru_2(x)[0]
        # x = torch.cat([x1r,x2r,x3r,x4r,x5r,x6r,x7r,x8r],dim=2)
        # x = self.gru_bn(x)
        # x = x1r + x2r + x3r + x4r + x5r + x6r + x7r + x8r
        # print('x',x.shape)
        # print('into GRU',x3.shape)
        # x4 = self.gru(x4)[0]
        # x3 = self.gru(x3)[0]
        # x2 = self.gru(x2)[0]

        # x = self.gru(x)[0]
        # x = self.gru2(x)[0]

        # print('out GRU',x3.shape)
        # x4 = x4.permute(1, 2, 0).contiguous()
        # x3 = x3.permute(1, 2, 0).contiguous()
        # x2 = x2.permute(1, 2, 0).contiguous()
        # x1 = x1.permute(1, 2, 0).contiguous()
        x = x.permute(1,2,0).contiguous()
        # print('5-1',x1.shape)

        # x4 = x4.view(x4.size(0), -1)
        # x3 = x3.view(x3.size(0), -1)
        # x2 = x2.view(x2.size(0), -1)
        # x1 = x1.view(x1.size(0), -1)
        x = x.view(x.size(0),-1)

        # x = x4 + x3 + x2 + x1
        # # w1 = x1 / x
        # # w2 = x2 / x
        # # w3 = x3 / x
        # # w4 = x4 / x
        # x = 0.35*x1 + 0.35*x2 + 0.15*x3 +0.15*x4
        # # x = w1*x1 + w2*x2 + w3*x3 + w4*x4
        # print('into gru_bn', x.shape)
        # x = self.gru_bn_2(x)
        x = self.gru_bn_3(x)
        # x = self.gru_bn2(x)
        x = self.relu(x)
        # x = self.tanh(x)
        # x = self.elu(x)
        # x =self.prelu(x)
        # print('into fc',x.shape)
        x = self.dropout(x)
        # x = self.fc_2(x)
        x_class = self.fc_3(x)
        x_rec = self.rec_3(x)
        # x = self.softmax(x)
        # print(x[0,:].cpu().detach().numpy())
        # plt.plot(x[0,:].cpu().detach().numpy())
        # plt.show()
        # plt.grid(linewidth=0.5, color='black')
        # plt.title('Real situation in one patch', fontdict={'size': 40})
        # plt.xlabel('Band Numbers', fontdict={'size': 40}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 40})
        # plt.xticks(fontsize=35)
        # plt.yticks(fontsize=35)
        # bwith = 2  # 边框宽度设置为2
        # TK = plt.gca()  # 获取边框
        # TK.spines['bottom'].set_linewidth(bwith)  # 图框下边
        # TK.spines['left'].set_linewidth(bwith)  # 图框左边
        # TK.spines['top'].set_linewidth(bwith)  # 图框上边
        # TK.spines['right'].set_linewidth(bwith)  # 图框右边
        # plt.show()
        # x = self.fc2(x)
        return x_class, x_rec

class zhoumultiscanning(nn.Module):

    @staticmethod
    def weight_init(m):
        # All weight matrices in our RNN and bias vectors are initialized with a uniform distribution, and the values of these weight matrices and bias vectors are initialized in the range [−0.1,0.1]
        if isinstance(m, (nn.Linear, nn.GRU, nn.Conv3d,nn.LSTM,nn.Conv2d)):
            init.kaiming_uniform_(m.weight.data)
            init.kaiming_uniform_(m.bias.data)

    def __init__(self, input_channels, n_classes, patch_size=5):
        # The proposed network model uses a single recurrent layer that adopts our modified GRUs of size 64 with sigmoid gate activation and PRetanh activation functions for hidden representations
        super(zhoumultiscanning, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.gru = nn.GRU(patch_size**2 , patch_size**2, 1, bidirectional=False, batch_first=False)  # TODO: try to change this ?
        self.gru_2 = nn.GRU(input_channels, 64, 1, bidirectional=False)
        self.conv1d = nn.Conv1d(in_channels=25,out_channels=25,kernel_size=3,stride=1)
        self.model_1d_cnn = nn.Conv1d(in_channels=patch_size**2,out_channels=128,kernel_size=3,stride=1)
        self.bn_1d_cnn = nn.BatchNorm1d(128 * 198)
        self.conv2d_1 = nn.Conv2d(in_channels=input_channels,out_channels=input_channels,kernel_size=2,stride=1)
        self.conv2d_2 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=2, stride=1)
        self.conv2d_3 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, stride=1)
        self.conv2d_4 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1)
        self.conv2d_5_1 = nn.Conv2d(in_channels=input_channels+1, out_channels=input_channels, kernel_size=1, stride=1)
        self.conv2d_5_2 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1)
        self.conv2d_5_3 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1)
        self.lstm_2 = nn.LSTM(input_channels, input_channels, 1, bidirectional=False)
        # self.vit = ViT(image_size=9,patch_size=3,num_classes=n_classes,depth=6,dim=n_classes,heads=6,mlp_dim=1024,channels=input_channels)
        # self.cait = CaiT(image_size=9,patch_size=3,num_classes=n_classes,depth=6,dim=n_classes,heads=6,mlp_dim=1024,cls_depth=1)
        # self.transformer_encoder = nn.TransformerEncoderLayer(d_model=input_channels,nhead=8)
        self.rnn_2 = nn.RNN(input_channels, 64, 1, bidirectional=False)
        self.gru_3 = nn.GRU(input_channels, patch_size**2, 1, bidirectional=True)
        self.gru_bn = nn.BatchNorm1d(patch_size**2 * (input_channels))
        self.gru_bn_trans = nn.BatchNorm1d(input_channels)
        self.gru_bn_2 = nn.BatchNorm1d((patch_size ** 2) * 64)
        self.gru_bn_3 = nn.BatchNorm1d(64)
        # self.layerNorm = nn.LayerNorm((patch_size ** 2) * input_channels)
        self.attention = nn.MultiheadAttention(embed_dim=input_channels,num_heads=1)
        self.attention_2 = nn.MultiheadAttention(embed_dim=64, num_heads=1)
        self.pos_embedding_spectral = nn.Parameter(torch.randn(1, input_channels + 1, 25))
        self.pos_embedding_spatial = nn.Parameter(torch.randn(1, (25) + 1, input_channels))
        self.cls_token_spectral = nn.Parameter(torch.randn(1, 1, 25))
        self.cls_token_spatial = nn.Parameter(torch.randn(1, 1, input_channels))
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.elu = nn.ELU()
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        self.ave1Dpooling = nn.AdaptiveAvgPool1d(input_channels)
        self.fc = nn.Linear(patch_size**2 * (input_channels), n_classes)
        self.fc_trans_1 = nn.Linear(input_channels, n_classes)
        self.fc_trans_2 = nn.Linear(input_channels, n_classes)
        self.fc_2 = nn.Linear((patch_size ** 2) * 64, n_classes)
        self.linearprojection_spatialtrans = nn.Linear(in_features=input_channels+1,out_features=input_channels)
        self.linearprojection_spectraltrans = nn.Linear(in_features=25, out_features=25)
        self.fc_3 = nn.Linear(64, n_classes)
        self.fc_for_1d_cnn = nn.Linear(128 * 198, n_classes)
        self.fc_vit = nn.Linear(input_channels,n_classes)
        self.softmax = nn.Softmax()


    def forward(self, x): #初始是第1方向
        print('x.shape',x.shape)

        x = x.squeeze(1) #B,F,P,P

        'horizontal'
        x1 = x
        x_horizontal = x1.reshape(x1.shape[0], x1.shape[1], -1)
        # print(x_horizontal)
        'vertical'
        x3 = torch.transpose(x1, 2, 3)
        x_vertical = x3.reshape(x3.shape[0], x3.shape[1], -1)
        # print(x_vertical)
        # 'diagonal'
        # xd1 = x[:, :, 0, 0]
        # xd1 = xd1.unsqueeze(2)
        # xd2 = x[:, :, 0, 1]
        # xd2 = xd2.unsqueeze(2)
        # xd3 = x[:, :, 1, 0]
        # xd3 = xd3.unsqueeze(2)
        # xd4 = x[:, :, 0, 2]
        # xd4 = xd4.unsqueeze(2)
        # xd5 = x[:, :, 1, 1]
        # xd5 = xd5.unsqueeze(2)
        # xd6 = x[:, :, 2, 0]
        # xd6 = xd6.unsqueeze(2)
        # xd7 = x[:, :, 0, 3]
        # xd7 = xd7.unsqueeze(2)
        # xd8 = x[:, :, 1, 2]
        # xd8 = xd8.unsqueeze(2)
        # xd9 = x[:, :, 2, 1]
        # xd9 = xd9.unsqueeze(2)
        # xd10 = x[:, :, 3, 0]
        # xd10 = xd10.unsqueeze(2)
        # xd11 = x[:, :, 0, 4]
        # xd11 = xd11.unsqueeze(2)
        # xd12 = x[:, :, 1, 3]
        # xd12 = xd12.unsqueeze(2)
        # xd13 = x[:, :, 2, 2]
        # xd13 = xd13.unsqueeze(2)
        # xd14 = x[:, :, 3, 1]
        # xd14 = xd14.unsqueeze(2)
        # xd15 = x[:, :, 4, 0]
        # xd15 = xd15.unsqueeze(2)
        # xd16 = x[:, :, 1, 4]
        # xd16 = xd16.unsqueeze(2)
        # xd17 = x[:, :, 2, 3]
        # xd17 = xd17.unsqueeze(2)
        # xd18 = x[:, :, 3, 2]
        # xd18 = xd18.unsqueeze(2)
        # xd19 = x[:, :, 4, 1]
        # xd19 = xd19.unsqueeze(2)
        # xd20 = x[:, :, 2, 4]
        # xd20 = xd20.unsqueeze(2)
        # xd21 = x[:, :, 3, 3]
        # xd21 = xd21.unsqueeze(2)
        # xd22 = x[:, :, 4, 2]
        # xd22 = xd22.unsqueeze(2)
        # xd23 = x[:, :, 3, 4]
        # xd23 = xd23.unsqueeze(2)
        # xd24 = x[:, :, 4, 3]
        # xd24 = xd24.unsqueeze(2)
        # xd25 = x[:, :, 4, 4]
        # xd25 = xd25.unsqueeze(2)
        # x_diagonal = torch.cat([xd1, xd2, xd3, xd4, xd5, xd6, xd7, xd8, xd9, xd10, xd11, xd12, xd13,
        #                         xd14, xd15, xd16, xd17, xd18, xd19, xd20, xd21, xd22, xd23, xd24, xd25], dim=2)
        # # print(x_diagonal)
        # 'zig-zag'
        # x1_d1 = x[:, :, 0, 0]
        # x1_d1 = x1_d1.unsqueeze(2)
        # # print(x1_d1.shape)
        # x2_d1 = x[:, :, 0, 1]
        # x2_d1 = x2_d1.unsqueeze(2)
        # x3_d1 = x[:, :, 1, 0]
        # x3_d1 = x3_d1.unsqueeze(2)
        # x4_d1 = x[:, :, 2, 0]
        # x4_d1 = x4_d1.unsqueeze(2)
        # x5_d1 = x[:, :, 1, 1]
        # x5_d1 = x5_d1.unsqueeze(2)
        # x6_d1 = x[:, :, 0, 2]
        # x6_d1 = x6_d1.unsqueeze(2)
        # x7_d1 = x[:, :, 0, 3]
        # x7_d1 = x7_d1.unsqueeze(2)
        # x8_d1 = x[:, :, 1, 2]
        # x8_d1 = x8_d1.unsqueeze(2)
        # x9_d1 = x[:, :, 2, 1]
        # x9_d1 = x9_d1.unsqueeze(2)
        # x10_d1 = x[:, :, 3, 0]
        # x10_d1 = x10_d1.unsqueeze(2)
        # x11_d1 = x[:, :, 4, 0]
        # x11_d1 = x11_d1.unsqueeze(2)
        # x12_d1 = x[:, :, 3, 1]
        # x12_d1 = x12_d1.unsqueeze(2)
        # x13_d1 = x[:, :, 2, 2]
        # x13_d1 = x13_d1.unsqueeze(2)
        # x14_d1 = x[:, :, 1, 3]
        # x14_d1 = x14_d1.unsqueeze(2)
        # x15_d1 = x[:, :, 0, 4]
        # x15_d1 = x15_d1.unsqueeze(2)
        # x16_d1 = x[:, :, 1, 4]
        # x16_d1 = x16_d1.unsqueeze(2)
        # x17_d1 = x[:, :, 2, 3]
        # x17_d1 = x17_d1.unsqueeze(2)
        # x18_d1 = x[:, :, 3, 2]
        # x18_d1 = x18_d1.unsqueeze(2)
        # x19_d1 = x[:, :, 4, 1]
        # x19_d1 = x19_d1.unsqueeze(2)
        # x20_d1 = x[:, :, 4, 2]
        # x20_d1 = x20_d1.unsqueeze(2)
        # x21_d1 = x[:, :, 3, 3]
        # x21_d1 = x21_d1.unsqueeze(2)
        # x22_d1 = x[:, :, 2, 4]
        # x22_d1 = x22_d1.unsqueeze(2)
        # x23_d1 = x[:, :, 3, 4]
        # x23_d1 = x23_d1.unsqueeze(2)
        # x24_d1 = x[:, :, 4, 3]
        # x24_d1 = x24_d1.unsqueeze(2)
        # x25_d1 = x[:, :, 4, 4]
        # x25_d1 = x25_d1.unsqueeze(2)
        # x_zigzag = torch.cat([x1_d1, x2_d1, x3_d1, x4_d1, x5_d1, x6_d1, x7_d1, x8_d1, x9_d1, x10_d1,
        #                       x11_d1, x12_d1, x13_d1, x14_d1, x15_d1, x16_d1, x17_d1, x18_d1, x19_d1, x20_d1, x21_d1,
        #                       x22_d1, x23_d1, x24_d1, x25_d1], dim=2)
        # # print(x_zigzag)
        #
        # 'perimeter'
        # xp1 = x[:, :, 0, 0]
        # xp1 = xp1.unsqueeze(2)
        # xp2 = x[:, :, 0, 1]
        # xp2 = xp2.unsqueeze(2)
        # xp3 = x[:, :, 0, 2]
        # xp3 = xp3.unsqueeze(2)
        # xp4 = x[:, :, 0, 3]
        # xp4 = xp4.unsqueeze(2)
        # xp5 = x[:, :, 0, 4]
        # xp5 = xp5.unsqueeze(2)
        # xp6 = x[:, :, 1, 4]
        # xp6 = xp6.unsqueeze(2)
        # xp7 = x[:, :, 2, 4]
        # xp7 = xp7.unsqueeze(2)
        # xp8 = x[:, :, 3, 4]
        # xp8 = xp8.unsqueeze(2)
        # xp9 = x[:, :, 4, 4]
        # xp9 = xp9.unsqueeze(2)
        # xp10 = x[:, :, 4, 3]
        # xp10 = xp10.unsqueeze(2)
        # xp11 = x[:, :, 4, 2]
        # xp11 = xp11.unsqueeze(2)
        # xp12 = x[:, :, 4, 1]
        # xp12 = xp12.unsqueeze(2)
        # xp13 = x[:, :, 4, 0]
        # xp13 = xp13.unsqueeze(2)
        # xp14 = x[:, :, 3, 0]
        # xp14 = xp14.unsqueeze(2)
        # xp15 = x[:, :, 2, 0]
        # xp15 = xp15.unsqueeze(2)
        # xp16 = x[:, :, 1, 0]
        # xp16 = xp16.unsqueeze(2)
        # xp17 = x[:, :, 1, 1]
        # xp17 = xp17.unsqueeze(2)
        # xp18 = x[:, :, 1, 2]
        # xp18 = xp18.unsqueeze(2)
        # xp19 = x[:, :, 1, 3]
        # xp19 = xp19.unsqueeze(2)
        # xp20 = x[:, :, 2, 3]
        # xp20 = xp20.unsqueeze(2)
        # xp21 = x[:, :, 3, 3]
        # xp21 = xp21.unsqueeze(2)
        # xp22 = x[:, :, 3, 2]
        # xp22 = xp22.unsqueeze(2)
        # xp23 = x[:, :, 3, 1]
        # xp23 = xp23.unsqueeze(2)
        # xp24 = x[:, :, 2, 1]
        # xp24 = xp24.unsqueeze(2)
        # xp25 = x[:, :, 2, 2]
        # xp25 = xp25.unsqueeze(2)
        # x_perimeter = torch.cat([xp1, xp2, xp3, xp4, xp5, xp6, xp7, xp8, xp9, xp10, xp11, xp12, xp13,
        #                          xp14, xp15, xp16, xp17, xp18, xp19, xp20, xp21, xp22, xp23, xp24, xp25], dim=2)
        # # print(x_perimeter)
        # 'expansion'
        # x_expansion = torch.flip(x_perimeter, [2])
        # # print(x_expansion)
        # 'hilbert'
        # xh1 = x[:, :, 0, 0]
        # xh1 = xh1.unsqueeze(2)
        # xh2 = x[:, :, 0, 1]
        # xh2 = xh2.unsqueeze(2)
        # xh3 = x[:, :, 0, 2]
        # xh3 = xh3.unsqueeze(2)
        # xh4 = x[:, :, 1, 2]
        # xh4 = xh4.unsqueeze(2)
        # xh5 = x[:, :, 1, 1]
        # xh5 = xh5.unsqueeze(2)
        # xh6 = x[:, :, 1, 0]
        # xh6 = xh6.unsqueeze(2)
        # xh7 = x[:, :, 2, 0]
        # xh7 = xh7.unsqueeze(2)
        # xh8 = x[:, :, 3, 0]
        # xh8 = xh8.unsqueeze(2)
        # xh9 = x[:, :, 4, 0]
        # xh9 = xh9.unsqueeze(2)
        # xh10 = x[:, :, 4, 1]
        # xh10 = xh10.unsqueeze(2)
        # xh11 = x[:, :, 3, 1]
        # xh11 = xh11.unsqueeze(2)
        # xh12 = x[:, :, 2, 1]
        # xh12 = xh12.unsqueeze(2)
        # xh13 = x[:, :, 2, 2]
        # xh13 = xh13.unsqueeze(2)
        # xh14 = x[:, :, 3, 2]
        # xh14 = xh14.unsqueeze(2)
        # xh15 = x[:, :, 4, 2]
        # xh15 = xh15.unsqueeze(2)
        # xh16 = x[:, :, 4, 3]
        # xh16 = xh16.unsqueeze(2)
        # xh17 = x[:, :, 4, 4]
        # xh17 = xh17.unsqueeze(2)
        # xh18 = x[:, :, 3, 4]
        # xh18 = xh18.unsqueeze(2)
        # xh19 = x[:, :, 3, 3]
        # xh19 = xh19.unsqueeze(2)
        # xh20 = x[:, :, 2, 3]
        # xh20 = xh20.unsqueeze(2)
        # xh21 = x[:, :, 2, 4]
        # xh21 = xh21.unsqueeze(2)
        # xh22 = x[:, :, 1, 4]
        # xh22 = xh22.unsqueeze(2)
        # xh23 = x[:, :, 1, 3]
        # xh23 = xh23.unsqueeze(2)
        # xh24 = x[:, :, 0, 3]
        # xh24 = xh24.unsqueeze(2)
        # xh25 = x[:, :, 0, 4]
        # xh25 = xh25.unsqueeze(2)
        # x_hilbert = torch.cat([xh1, xh2, xh3, xh4, xh5, xh6, xh7, xh8, xh9, xh10, xh11, xh12, xh13,
        #                        xh14, xh15, xh16, xh17, xh18, xh19, xh20, xh21, xh22, xh23, xh24, xh25], dim=2)
        # # print(x_hilbert)
        # 'U-turn'
        x1_0 = x[:, :, 0, :]
        x1_1 = x[:, :, 1, :]
        x1_2 = x[:, :, 2, :]
        x1_3 = x[:, :, 3, :]
        x1_4 = x[:, :, 4, :]
        x1_1f = torch.flip(x1_1, [2])
        x1_3f = torch.flip(x1_3, [2])
        x_uturn = torch.cat([x1_0, x1_1f, x1_2, x1_3f, x1_4], dim=2)
        # # print(x_uturn)

        # x_vertical = x_vertical.permute(2, 0, 1) #(25,16,103)
        # x_horizontal = x_horizontal.permute(2, 0, 1)
        # x_diagonal = x_diagonal.permute(2, 0, 1)
        # x_zigzag = x_zigzag.permute(2, 0, 1)
        # x_expansion = x_expansion.permute(2, 0, 1)
        # x_perimeter = x_perimeter.permute(2, 0, 1)
        # x_hilbert = x_hilbert.permute(2, 0, 1)
        x_uturn = x_uturn.permute(2, 0, 1)


        '改变scanning的模式'
        x = x_uturn #(X,B,C)
        plt.subplot(411)
        plt.imshow(x[:, 0, :].cpu().detach().numpy(), cmap='gray')
        print('x.sequence',x.shape) #(seq_len,batch,feature_dimension)

        # plt.show()
        # #(revision)spectral-spatial 1-D CNN
        # x_for_1dcnn_in = x.permute(1,0,2)
        # # model_1d_cnn = nn.Conv1d(in_channels=x_for_1dcnn_in.size(1),out_channels=64,kernel_size=3,stride=2).to(device = 'cuda')
        # x_for_1dcnn_out = self.model_1d_cnn(x_for_1dcnn_in)
        # print('x_for_id_cnn_out',x_for_1dcnn_out.shape)
        # x_for_1dcnn_out = x_for_1dcnn_out.view(x_for_1dcnn_out.size(0), -1)
        # x_for_1dcnn_out = self.bn_1d_cnn(x_for_1dcnn_out)
        # x_for_1dcnn_out = self.relu(x_for_1dcnn_out)
        # x_for_1dcnn_out = self.dropout(x_for_1dcnn_out)
        # x_for_1dcnn_out_preds = self.fc_for_1d_cnn(x_for_1dcnn_out)

        #LSTM
        print('x.shape',x.shape) #(squ_len,batch_size,feature_dimension)
        h0 = torch.zeros(1, x.size(1), x.size(2)).to(device='cuda')
        print('h0.shape',h0.shape)
        c0 = torch.zeros(1, x.size(1), x.size(2)).to(device="cuda")
        out, (hn,cn) = self.lstm_2(x,(h0,c0))
        print('out.shape',out.shape) #(25,100,200)
        plt.imshow(out[0,:,:].cpu().detach().numpy(),cmap='gray')
        # plt.show()
        x = out.permute(1,2,0).contiguous()
        print('x_out.shape',x.shape) #(batch_size,feature_dim,squ_len) (100,200,25)
        x = x.view(x.size(0), -1)

        #batchnorm+fc 分类工作
        x = self.gru_bn(x)
        x = self.prelu(x)
        x = self.dropout(x)
        preds = self.fc(x)

        # print('preds',x.shape)

        return preds

class zhoumultiscanning_Trans(nn.Module):

    @staticmethod
    def weight_init(m):
        # All weight matrices in our RNN and bias vectors are initialized with a uniform distribution, and the values of these weight matrices and bias vectors are initialized in the range [−0.1,0.1]
        if isinstance(m, (nn.Linear, nn.GRU, nn.Conv3d,nn.LSTM,nn.Conv2d)):
            init.kaiming_uniform_(m.weight.data)
            init.kaiming_uniform_(m.bias.data)

    def __init__(self, input_channels, n_classes, patch_size=5):
        # The proposed network model uses a single recurrent layer that adopts our modified GRUs of size 64 with sigmoid gate activation and PRetanh activation functions for hidden representations
        super(zhoumultiscanning_Trans, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.gru = nn.GRU(patch_size**2 , patch_size**2, 1, bidirectional=False, batch_first=False)  # TODO: try to change this ?
        self.gru_2 = nn.GRU(input_channels, 64, 1, bidirectional=False)
        self.conv1d = nn.Conv1d(in_channels=25,out_channels=25,kernel_size=3,stride=1)
        self.model_1d_cnn = nn.Conv1d(in_channels=patch_size**2,out_channels=128,kernel_size=3,stride=1)
        self.bn_1d_cnn = nn.BatchNorm1d(128 * 198)
        self.conv2d_1 = nn.Conv2d(in_channels=input_channels,out_channels=input_channels,kernel_size=2,stride=1)
        self.conv2d_2 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=2, stride=1)
        self.conv2d_3 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, stride=1)
        self.conv2d_4 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1)
        self.conv2d_5_1 = nn.Conv2d(in_channels=input_channels+1, out_channels=input_channels, kernel_size=1, stride=1)
        self.conv2d_5_2 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1)
        self.conv2d_5_3 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1)
        self.lstm_2 = nn.LSTM(input_channels, input_channels, 1, bidirectional=False)
        # self.vit = ViT(image_size=9,patch_size=3,num_classes=n_classes,depth=6,dim=n_classes,heads=6,mlp_dim=1024,channels=input_channels)
        # self.cait = CaiT(image_size=9,patch_size=3,num_classes=n_classes,depth=6,dim=n_classes,heads=6,mlp_dim=1024,cls_depth=1)
        # self.transformer_encoder = nn.TransformerEncoderLayer(d_model=input_channels,nhead=8)
        self.rnn_2 = nn.RNN(input_channels, 64, 1, bidirectional=False)
        self.gru_3 = nn.GRU(input_channels, patch_size**2, 1, bidirectional=True)
        self.gru_bn = nn.BatchNorm1d(patch_size**2 * (input_channels))
        self.gru_bn_trans = nn.BatchNorm1d(input_channels)
        self.gru_bn_2 = nn.BatchNorm1d((patch_size ** 2) * 64)
        self.gru_bn_3 = nn.BatchNorm1d(64)
        # self.layerNorm = nn.LayerNorm((patch_size ** 2) * input_channels)
        self.attention = nn.MultiheadAttention(embed_dim=input_channels,num_heads=1)
        self.attention_2 = nn.MultiheadAttention(embed_dim=64, num_heads=1)
        self.pos_embedding_spectral = nn.Parameter(torch.randn(1, input_channels + 1, 25))
        self.pos_embedding_spatial = nn.Parameter(torch.randn(1, (25) + 1, input_channels))
        self.cls_token_spectral = nn.Parameter(torch.randn(1, 1, 25))
        self.cls_token_spatial = nn.Parameter(torch.randn(1, 1, input_channels))
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.elu = nn.ELU()
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        self.ave1Dpooling = nn.AdaptiveAvgPool1d(input_channels)
        self.fc = nn.Linear(patch_size**2 * (input_channels), n_classes)
        self.fc_trans_1 = nn.Linear(input_channels, n_classes)
        self.fc_trans_2 = nn.Linear(input_channels, n_classes)
        self.fc_2 = nn.Linear((patch_size ** 2) * 64, n_classes)
        self.linearprojection_spatialtrans = nn.Linear(in_features=input_channels+1,out_features=input_channels)
        self.linearprojection_spectraltrans = nn.Linear(in_features=25, out_features=25)
        self.fc_3 = nn.Linear(64, n_classes)
        self.fc_for_1d_cnn = nn.Linear(128 * 198, n_classes)
        self.fc_vit = nn.Linear(input_channels,n_classes)
        self.softmax = nn.Softmax()


    def forward(self, x): #初始是第1方向
        print('x.shape',x.shape)

        x = x.squeeze(1) #B,F,P,P

        'horizontal'
        x1 = x
        x_horizontal = x1.reshape(x1.shape[0], x1.shape[1], -1)
        # print(x_horizontal)
        'vertical'
        x3 = torch.transpose(x1, 2, 3)
        x_vertical = x3.reshape(x3.shape[0], x3.shape[1], -1)
        # print(x_vertical)
        # 'diagonal'
        # xd1 = x[:, :, 0, 0]
        # xd1 = xd1.unsqueeze(2)
        # xd2 = x[:, :, 0, 1]
        # xd2 = xd2.unsqueeze(2)
        # xd3 = x[:, :, 1, 0]
        # xd3 = xd3.unsqueeze(2)
        # xd4 = x[:, :, 0, 2]
        # xd4 = xd4.unsqueeze(2)
        # xd5 = x[:, :, 1, 1]
        # xd5 = xd5.unsqueeze(2)
        # xd6 = x[:, :, 2, 0]
        # xd6 = xd6.unsqueeze(2)
        # xd7 = x[:, :, 0, 3]
        # xd7 = xd7.unsqueeze(2)
        # xd8 = x[:, :, 1, 2]
        # xd8 = xd8.unsqueeze(2)
        # xd9 = x[:, :, 2, 1]
        # xd9 = xd9.unsqueeze(2)
        # xd10 = x[:, :, 3, 0]
        # xd10 = xd10.unsqueeze(2)
        # xd11 = x[:, :, 0, 4]
        # xd11 = xd11.unsqueeze(2)
        # xd12 = x[:, :, 1, 3]
        # xd12 = xd12.unsqueeze(2)
        # xd13 = x[:, :, 2, 2]
        # xd13 = xd13.unsqueeze(2)
        # xd14 = x[:, :, 3, 1]
        # xd14 = xd14.unsqueeze(2)
        # xd15 = x[:, :, 4, 0]
        # xd15 = xd15.unsqueeze(2)
        # xd16 = x[:, :, 1, 4]
        # xd16 = xd16.unsqueeze(2)
        # xd17 = x[:, :, 2, 3]
        # xd17 = xd17.unsqueeze(2)
        # xd18 = x[:, :, 3, 2]
        # xd18 = xd18.unsqueeze(2)
        # xd19 = x[:, :, 4, 1]
        # xd19 = xd19.unsqueeze(2)
        # xd20 = x[:, :, 2, 4]
        # xd20 = xd20.unsqueeze(2)
        # xd21 = x[:, :, 3, 3]
        # xd21 = xd21.unsqueeze(2)
        # xd22 = x[:, :, 4, 2]
        # xd22 = xd22.unsqueeze(2)
        # xd23 = x[:, :, 3, 4]
        # xd23 = xd23.unsqueeze(2)
        # xd24 = x[:, :, 4, 3]
        # xd24 = xd24.unsqueeze(2)
        # xd25 = x[:, :, 4, 4]
        # xd25 = xd25.unsqueeze(2)
        # x_diagonal = torch.cat([xd1, xd2, xd3, xd4, xd5, xd6, xd7, xd8, xd9, xd10, xd11, xd12, xd13,
        #                         xd14, xd15, xd16, xd17, xd18, xd19, xd20, xd21, xd22, xd23, xd24, xd25], dim=2)
        # # print(x_diagonal)
        # 'zig-zag'
        # x1_d1 = x[:, :, 0, 0]
        # x1_d1 = x1_d1.unsqueeze(2)
        # # print(x1_d1.shape)
        # x2_d1 = x[:, :, 0, 1]
        # x2_d1 = x2_d1.unsqueeze(2)
        # x3_d1 = x[:, :, 1, 0]
        # x3_d1 = x3_d1.unsqueeze(2)
        # x4_d1 = x[:, :, 2, 0]
        # x4_d1 = x4_d1.unsqueeze(2)
        # x5_d1 = x[:, :, 1, 1]
        # x5_d1 = x5_d1.unsqueeze(2)
        # x6_d1 = x[:, :, 0, 2]
        # x6_d1 = x6_d1.unsqueeze(2)
        # x7_d1 = x[:, :, 0, 3]
        # x7_d1 = x7_d1.unsqueeze(2)
        # x8_d1 = x[:, :, 1, 2]
        # x8_d1 = x8_d1.unsqueeze(2)
        # x9_d1 = x[:, :, 2, 1]
        # x9_d1 = x9_d1.unsqueeze(2)
        # x10_d1 = x[:, :, 3, 0]
        # x10_d1 = x10_d1.unsqueeze(2)
        # x11_d1 = x[:, :, 4, 0]
        # x11_d1 = x11_d1.unsqueeze(2)
        # x12_d1 = x[:, :, 3, 1]
        # x12_d1 = x12_d1.unsqueeze(2)
        # x13_d1 = x[:, :, 2, 2]
        # x13_d1 = x13_d1.unsqueeze(2)
        # x14_d1 = x[:, :, 1, 3]
        # x14_d1 = x14_d1.unsqueeze(2)
        # x15_d1 = x[:, :, 0, 4]
        # x15_d1 = x15_d1.unsqueeze(2)
        # x16_d1 = x[:, :, 1, 4]
        # x16_d1 = x16_d1.unsqueeze(2)
        # x17_d1 = x[:, :, 2, 3]
        # x17_d1 = x17_d1.unsqueeze(2)
        # x18_d1 = x[:, :, 3, 2]
        # x18_d1 = x18_d1.unsqueeze(2)
        # x19_d1 = x[:, :, 4, 1]
        # x19_d1 = x19_d1.unsqueeze(2)
        # x20_d1 = x[:, :, 4, 2]
        # x20_d1 = x20_d1.unsqueeze(2)
        # x21_d1 = x[:, :, 3, 3]
        # x21_d1 = x21_d1.unsqueeze(2)
        # x22_d1 = x[:, :, 2, 4]
        # x22_d1 = x22_d1.unsqueeze(2)
        # x23_d1 = x[:, :, 3, 4]
        # x23_d1 = x23_d1.unsqueeze(2)
        # x24_d1 = x[:, :, 4, 3]
        # x24_d1 = x24_d1.unsqueeze(2)
        # x25_d1 = x[:, :, 4, 4]
        # x25_d1 = x25_d1.unsqueeze(2)
        # x_zigzag = torch.cat([x1_d1, x2_d1, x3_d1, x4_d1, x5_d1, x6_d1, x7_d1, x8_d1, x9_d1, x10_d1,
        #                       x11_d1, x12_d1, x13_d1, x14_d1, x15_d1, x16_d1, x17_d1, x18_d1, x19_d1, x20_d1, x21_d1,
        #                       x22_d1, x23_d1, x24_d1, x25_d1], dim=2)
        # # print(x_zigzag)
        #
        # 'perimeter'
        # xp1 = x[:, :, 0, 0]
        # xp1 = xp1.unsqueeze(2)
        # xp2 = x[:, :, 0, 1]
        # xp2 = xp2.unsqueeze(2)
        # xp3 = x[:, :, 0, 2]
        # xp3 = xp3.unsqueeze(2)
        # xp4 = x[:, :, 0, 3]
        # xp4 = xp4.unsqueeze(2)
        # xp5 = x[:, :, 0, 4]
        # xp5 = xp5.unsqueeze(2)
        # xp6 = x[:, :, 1, 4]
        # xp6 = xp6.unsqueeze(2)
        # xp7 = x[:, :, 2, 4]
        # xp7 = xp7.unsqueeze(2)
        # xp8 = x[:, :, 3, 4]
        # xp8 = xp8.unsqueeze(2)
        # xp9 = x[:, :, 4, 4]
        # xp9 = xp9.unsqueeze(2)
        # xp10 = x[:, :, 4, 3]
        # xp10 = xp10.unsqueeze(2)
        # xp11 = x[:, :, 4, 2]
        # xp11 = xp11.unsqueeze(2)
        # xp12 = x[:, :, 4, 1]
        # xp12 = xp12.unsqueeze(2)
        # xp13 = x[:, :, 4, 0]
        # xp13 = xp13.unsqueeze(2)
        # xp14 = x[:, :, 3, 0]
        # xp14 = xp14.unsqueeze(2)
        # xp15 = x[:, :, 2, 0]
        # xp15 = xp15.unsqueeze(2)
        # xp16 = x[:, :, 1, 0]
        # xp16 = xp16.unsqueeze(2)
        # xp17 = x[:, :, 1, 1]
        # xp17 = xp17.unsqueeze(2)
        # xp18 = x[:, :, 1, 2]
        # xp18 = xp18.unsqueeze(2)
        # xp19 = x[:, :, 1, 3]
        # xp19 = xp19.unsqueeze(2)
        # xp20 = x[:, :, 2, 3]
        # xp20 = xp20.unsqueeze(2)
        # xp21 = x[:, :, 3, 3]
        # xp21 = xp21.unsqueeze(2)
        # xp22 = x[:, :, 3, 2]
        # xp22 = xp22.unsqueeze(2)
        # xp23 = x[:, :, 3, 1]
        # xp23 = xp23.unsqueeze(2)
        # xp24 = x[:, :, 2, 1]
        # xp24 = xp24.unsqueeze(2)
        # xp25 = x[:, :, 2, 2]
        # xp25 = xp25.unsqueeze(2)
        # x_perimeter = torch.cat([xp1, xp2, xp3, xp4, xp5, xp6, xp7, xp8, xp9, xp10, xp11, xp12, xp13,
        #                          xp14, xp15, xp16, xp17, xp18, xp19, xp20, xp21, xp22, xp23, xp24, xp25], dim=2)
        # # print(x_perimeter)
        # 'expansion'
        # x_expansion = torch.flip(x_perimeter, [2])
        # # print(x_expansion)
        # 'hilbert'
        # xh1 = x[:, :, 0, 0]
        # xh1 = xh1.unsqueeze(2)
        # xh2 = x[:, :, 0, 1]
        # xh2 = xh2.unsqueeze(2)
        # xh3 = x[:, :, 0, 2]
        # xh3 = xh3.unsqueeze(2)
        # xh4 = x[:, :, 1, 2]
        # xh4 = xh4.unsqueeze(2)
        # xh5 = x[:, :, 1, 1]
        # xh5 = xh5.unsqueeze(2)
        # xh6 = x[:, :, 1, 0]
        # xh6 = xh6.unsqueeze(2)
        # xh7 = x[:, :, 2, 0]
        # xh7 = xh7.unsqueeze(2)
        # xh8 = x[:, :, 3, 0]
        # xh8 = xh8.unsqueeze(2)
        # xh9 = x[:, :, 4, 0]
        # xh9 = xh9.unsqueeze(2)
        # xh10 = x[:, :, 4, 1]
        # xh10 = xh10.unsqueeze(2)
        # xh11 = x[:, :, 3, 1]
        # xh11 = xh11.unsqueeze(2)
        # xh12 = x[:, :, 2, 1]
        # xh12 = xh12.unsqueeze(2)
        # xh13 = x[:, :, 2, 2]
        # xh13 = xh13.unsqueeze(2)
        # xh14 = x[:, :, 3, 2]
        # xh14 = xh14.unsqueeze(2)
        # xh15 = x[:, :, 4, 2]
        # xh15 = xh15.unsqueeze(2)
        # xh16 = x[:, :, 4, 3]
        # xh16 = xh16.unsqueeze(2)
        # xh17 = x[:, :, 4, 4]
        # xh17 = xh17.unsqueeze(2)
        # xh18 = x[:, :, 3, 4]
        # xh18 = xh18.unsqueeze(2)
        # xh19 = x[:, :, 3, 3]
        # xh19 = xh19.unsqueeze(2)
        # xh20 = x[:, :, 2, 3]
        # xh20 = xh20.unsqueeze(2)
        # xh21 = x[:, :, 2, 4]
        # xh21 = xh21.unsqueeze(2)
        # xh22 = x[:, :, 1, 4]
        # xh22 = xh22.unsqueeze(2)
        # xh23 = x[:, :, 1, 3]
        # xh23 = xh23.unsqueeze(2)
        # xh24 = x[:, :, 0, 3]
        # xh24 = xh24.unsqueeze(2)
        # xh25 = x[:, :, 0, 4]
        # xh25 = xh25.unsqueeze(2)
        # x_hilbert = torch.cat([xh1, xh2, xh3, xh4, xh5, xh6, xh7, xh8, xh9, xh10, xh11, xh12, xh13,
        #                        xh14, xh15, xh16, xh17, xh18, xh19, xh20, xh21, xh22, xh23, xh24, xh25], dim=2)
        # # print(x_hilbert)
        # 'U-turn'
        x1_0 = x[:, :, 0, :]
        x1_1 = x[:, :, 1, :]
        x1_2 = x[:, :, 2, :]
        x1_3 = x[:, :, 3, :]
        x1_4 = x[:, :, 4, :]
        x1_1f = torch.flip(x1_1, [2])
        x1_3f = torch.flip(x1_3, [2])
        x_uturn = torch.cat([x1_0, x1_1f, x1_2, x1_3f, x1_4], dim=2)
        # # print(x_uturn)

        # x_vertical = x_vertical.permute(2, 0, 1) #(25,16,103)
        # x_horizontal = x_horizontal.permute(2, 0, 1)
        # x_diagonal = x_diagonal.permute(2, 0, 1)
        # x_zigzag = x_zigzag.permute(2, 0, 1)
        # x_expansion = x_expansion.permute(2, 0, 1)
        # x_perimeter = x_perimeter.permute(2, 0, 1)
        # x_hilbert = x_hilbert.permute(2, 0, 1)
        x_uturn = x_uturn.permute(2, 0, 1)


        '改变scanning的模式'
        x = x_uturn #(X,B,C)
        plt.subplot(411)
        plt.imshow(x[:, 0, :].cpu().detach().numpy(), cmap='gray')
        print('x.sequence',x.shape) #(seq_len,batch,feature_dimension)

        # plt.show()
        # #(revision)spectral-spatial 1-D CNN
        # x_for_1dcnn_in = x.permute(1,0,2)
        # # model_1d_cnn = nn.Conv1d(in_channels=x_for_1dcnn_in.size(1),out_channels=64,kernel_size=3,stride=2).to(device = 'cuda')
        # x_for_1dcnn_out = self.model_1d_cnn(x_for_1dcnn_in)
        # print('x_for_id_cnn_out',x_for_1dcnn_out.shape)
        # x_for_1dcnn_out = x_for_1dcnn_out.view(x_for_1dcnn_out.size(0), -1)
        # x_for_1dcnn_out = self.bn_1d_cnn(x_for_1dcnn_out)
        # x_for_1dcnn_out = self.relu(x_for_1dcnn_out)
        # x_for_1dcnn_out = self.dropout(x_for_1dcnn_out)
        # x_for_1dcnn_out_preds = self.fc_for_1d_cnn(x_for_1dcnn_out)

        #LSTM
        print('x.shape',x.shape) #(squ_len,batch_size,feature_dimension)
        h0 = torch.zeros(1, x.size(1), x.size(2)).to(device='cuda')
        print('h0.shape',h0.shape)
        c0 = torch.zeros(1, x.size(1), x.size(2)).to(device="cuda")
        out, (hn,cn) = self.lstm_2(x,(h0,c0))
        print('out.shape',out.shape) #(25,100,200)
        plt.imshow(out[0,:,:].cpu().detach().numpy(),cmap='gray')
        # plt.show()
        x = out.permute(1,2,0).contiguous()
        print('x_out.shape',x.shape) #(batch_size,feature_dim,squ_len) (100,200,25)
        x = x.view(x.size(0), -1)

        #batchnorm+fc 分类工作
        x = self.gru_bn(x)
        x = self.prelu(x)
        x = self.dropout(x)
        preds = self.fc(x)

        # print('preds',x.shape)

        return preds

class zhouSSViT(nn.Module):

    @staticmethod
    def weight_init(m):
        # All weight matrices in our RNN and bias vectors are initialized with a uniform distribution, and the values of these weight matrices and bias vectors are initialized in the range [−0.1,0.1]
        if isinstance(m, (nn.Linear, nn.GRU, nn.Conv3d,nn.LSTM,nn.Conv2d)):
            init.kaiming_uniform_(m.weight.data)
            init.kaiming_uniform_(m.bias.data)

    def __init__(self, input_channels, n_classes, patch_size=9, pool = 'cls'):
        # The proposed network model uses a single recurrent layer that adopts our modified GRUs of size 64 with sigmoid gate activation and PRetanh activation functions for hidden representations
        super(zhouSSViT, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.gru = nn.GRU(patch_size**2 , patch_size**2, 1, bidirectional=False, batch_first=False)  # TODO: try to change this ?
        self.gru_2 = nn.GRU(input_channels, 64, 1, bidirectional=False)
        self.conv1d = nn.Conv1d(in_channels=25,out_channels=25,kernel_size=3,stride=1)
        self.conv2d_1 = nn.Conv2d(in_channels=input_channels,out_channels=input_channels,kernel_size=2,stride=1)
        self.conv2d_2 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=2, stride=1)
        self.conv2d_3 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, stride=1)
        self.conv2d_4 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1)
        self.conv2d_5_1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1)
        self.conv2d_5_2 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1)
        self.conv2d_5_3 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1)
        self.lstm_2 = nn.LSTM(input_channels, input_channels, 1, bidirectional=False)
        # self.vit = ViT(image_size=9,patch_size=3,num_classes=n_classes,depth=6,dim=n_classes,heads=6,mlp_dim=1024,channels=input_channels)
        # self.cait = CaiT(image_size=9,patch_size=3,num_classes=n_classes,depth=6,dim=n_classes,heads=6,mlp_dim=1024,cls_depth=1)
        # self.transformer_encoder = nn.TransformerEncoderLayer(d_model=input_channels,nhead=8)
        self.rnn_2 = nn.RNN(input_channels, 64, 1, bidirectional=False)
        self.gru_3 = nn.GRU(input_channels, patch_size**2, 1, bidirectional=True)
        self.gru_bn = nn.BatchNorm1d(patch_size**2 * (input_channels))
        self.gru_bn_trans = nn.BatchNorm1d(input_channels)
        self.gru_bn_2 = nn.BatchNorm1d((patch_size ** 2) * 64)
        self.gru_bn_spectralclasstoken = nn.BatchNorm1d(input_channels+1)
        # self.layerNorm = nn.LayerNorm((patch_size ** 2) * input_channels)
        self.attention = nn.MultiheadAttention(embed_dim=input_channels,num_heads=1)
        self.attention_2 = nn.MultiheadAttention(embed_dim=64, num_heads=1)
        self.pos_embedding_spectral = nn.Parameter(torch.randn(1, input_channels + 1, 25))
        self.pos_embedding_spatial = nn.Parameter(torch.randn(1, (25) + 1, input_channels))
        self.cls_token_spectral = nn.Parameter(torch.randn(1, 1, 25))
        self.cls_token_spatial = nn.Parameter(torch.randn(1, 1, input_channels))
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.elu = nn.ELU()
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        self.ave1Dpooling = nn.AdaptiveAvgPool1d(input_channels)
        self.ave1Dpooling_spectral = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(patch_size**2 * (input_channels), n_classes)
        self.fc_trans_1 = nn.Linear(input_channels, n_classes)
        self.fc_trans_2 = nn.Linear(input_channels+1, n_classes)
        self.dense_layer_spectral = nn.Linear(25, 25)
        self.dense_layer_spatial = nn.Linear(input_channels,input_channels)
        self.fc_2 = nn.Linear((patch_size ** 2) * 64, n_classes)
        self.linearprojection_spatialtrans = nn.Linear(in_features=input_channels+1,out_features=input_channels)
        self.linearprojection_spectraltrans = nn.Linear(in_features=25, out_features=25)
        self.linearprojection_25_to_81 = nn.Linear(in_features=25, out_features=81)
        self.fc_3 = nn.Linear(64, n_classes)
        self.fc_vit = nn.Linear(input_channels,n_classes)
        self.fc_joint = nn.Linear(n_classes*2,n_classes)
        self.softmax = nn.Softmax()
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(input_channels),
            nn.Linear(input_channels, n_classes)
        )
        self.pool = pool
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


    def forward(self, x): #初始是第1方向
        print('x.shape',x.shape)

        x = x.squeeze(1) #B,F,P,P

        #ResNet patch_size = 9 for SA PU
        x = self.conv2d_1(x)
        print('1',x.shape)
        x = self.prelu(x)
        x = self.conv2d_2(x)
        print('2',x.shape)
        x_res = self.prelu(x)
        x_res = self.conv2d_3(x_res)
        print('3',x.shape)
        x_res = self.prelu(x_res)
        x_res_res = self.conv2d_4(x_res)
        x_res_res = self.prelu(x_res_res)
        x =  x_res +x_res_res
        print('4',x.shape)

        # ResNet patch_size = 5 for IP dataset
        # x_spectraltrans_reconstruct = self.conv2d_5_1(x)
        # x_spectraltrans_reconstruct_1 = x_spectraltrans_reconstruct
        # x_spectraltrans_reconstruct_2 = self.relu(x_spectraltrans_reconstruct)
        # x_spectraltrans_reconstruct_2 = self.conv2d_5_2(x_spectraltrans_reconstruct_2)
        # x_spectraltrans_reconstruct_2 = self.relu(x_spectraltrans_reconstruct_2)
        # x_spectraltrans_reconstruct_3 = self.conv2d_5_3(x_spectraltrans_reconstruct_2)
        # x_spectraltrans_reconstruct_3 = self.relu(x_spectraltrans_reconstruct_2)
        # x = x_spectraltrans_reconstruct_1 + x_spectraltrans_reconstruct_2 + x_spectraltrans_reconstruct_3



        'horizontal'
        x1 = x
        x_horizontal = x1.reshape(x1.shape[0], x1.shape[1], -1)

        x_horizontal = x_horizontal.permute(2, 0, 1)


        '改变scanning的模式'
        x = x_horizontal #(X,B,C) (25,2,204)
        # x_show = x.permute(2,1,0).contiguous()
        # plt.subplot(121)
        # plt.plot(x[:, 0, :].cpu().detach().numpy())
        # plt.plot(x[:, 0, 0].cpu().detach().numpy(),linewidth=5)
        # plt.plot(x[:, 0, 203].cpu().detach().numpy(), linewidth=5)
        # plt.subplot(122)
        # plt.imshow(x[:, 0, :].cpu().detach().numpy())
        print('x.sequence',x.shape) #(seq_len,batch,feature_dimension)
        # plt.show()
        # plt.subplot(211)
        # plt.imshow(x[:, 0, :].cpu().detach().numpy(), cmap='gray')
        # plt.subplot(212)
        # plt.plot(x_show[:, 0, :].cpu().detach().numpy())
        # plt.show()

        #Transformerencoder 从这里开始
        #spectral trans (SpeViT)
        x_spectraltrans = x.permute(2,1,0).contiguous() #(C,B,X)
        x_spectraltrans = x_spectraltrans.permute(1, 0, 2).contiguous() #(B,C,X)
        x_spectraltrans = self.linearprojection_spectraltrans(x_spectraltrans)
        x_spectraltrans = x_spectraltrans.permute(1, 0, 2).contiguous()#(C,B,X)

        x_spectraltrans = x_spectraltrans.permute(1, 0, 2).contiguous()  # (B,C,X)
        b , n , _ = x_spectraltrans.shape
        cls_tokens_spectral =  repeat(self.cls_token_spectral, '() n d -> b n d', b = b)
        x_spectraltrans = torch.cat((cls_tokens_spectral, x_spectraltrans), dim=1)
        x_spectraltrans += self.pos_embedding_spectral[:, :(x_spectraltrans.size(1) + 1)]
        x_spectraltrans = x_spectraltrans.permute(1, 0, 2).contiguous()  # (C,B,X)
        print('222',x_spectraltrans.shape)

        encoder_layer_spectral = nn.TransformerEncoderLayer(d_model=25, nhead=1, dim_feedforward=32,activation='gelu').to(device='cuda')
        transformer_encoder_spectral = nn.TransformerEncoder(encoder_layer_spectral, num_layers=1, norm=None).to(device='cuda')

        x_spectraltrans_output = transformer_encoder_spectral(x_spectraltrans) #(C,B,X)(205,b,25)

        x_show_2 = x_spectraltrans_output.permute(2,1,0).contiguous() #(25, b, 205)

        x_pool = reduce(x_show_2, 'x b c -> x b', reduction='mean')
        print('x_pool',x_pool.shape)
        x_cls = x_show_2[:, :, 0]
        print('x_cls',x_cls.shape)
        plt.plot(x_show_2[:,0,:].cpu().detach().numpy())
        plt.plot(x_pool[:,0].cpu().detach().numpy(),color='blue', linewidth=5)
        plt.plot(x_cls[:, 0].cpu().detach().numpy(), color='red', linewidth=5)

        plt.grid(linewidth=0.5, color='black')
        plt.title('All tokens', fontdict={'size': 40})
        plt.xlabel('Spatial size', fontdict={'size': 40}, fontweight='bold')
        plt.ylabel('Values', fontdict={'size': 40})
        plt.xticks(fontsize=35)
        plt.yticks(fontsize=35)
        bwith = 2  # 边框宽度设置为2
        TK = plt.gca()  # 获取边框
        TK.spines['bottom'].set_linewidth(bwith)  # 图框下边
        TK.spines['left'].set_linewidth(bwith)  # 图框左边
        TK.spines['top'].set_linewidth(bwith)  # 图框上边
        TK.spines['right'].set_linewidth(bwith)  # 图框右边

        plt.show()

        x_spectraltrans_output = self.dense_layer_spectral(x_spectraltrans_output) #attention之后的全连接层
        x_spectraltrans_output = self.relu(x_spectraltrans_output)

        #SpeViT的output layer
        x_spectral_classtoken = self.ave1Dpooling_spectral(x_spectraltrans_output)

        print('spectraltoken',x_spectral_classtoken.shape) #(205,b,1)


        # x_spectral_classtoken = x_spectraltrans_output[:,:,0]
        x_spectral_classtoken = x_spectral_classtoken.permute(1, 0, 2).contiguous()
        x_spectral_classtoken = x_spectral_classtoken.view(x_spectral_classtoken.size(0), -1)
        x_spectral_classtoken = self.gru_bn_spectralclasstoken(x_spectral_classtoken)
        # x_spectral_classtoken = self.prelu(x_spectral_classtoken)
        x_spectral_classtoken = self.dropout(x_spectral_classtoken)
        preds_SpeViT = self.fc_trans_2(x_spectral_classtoken) #SpeViT de output
        # x_spectraltrans_output = x_spectraltrans_output.permute()
        # print('x_spectral_out', x_spectraltrans_output.shape) #(200+1,2,25)
        # x_spectraltrans_reconstruct = x_spectraltrans_output.reshape(x_spectraltrans_output.shape[1],x_spectraltrans_output.shape[0],5,5)

        #进入spatial trans (SpaViT)
        x_spatialtrans = x_spectraltrans_output.permute(2,1,0) #(x,b,c)
        print('111',x_spatialtrans.shape) #(25,2,205)
        # x_spatialtrans = self.gelu(x_spatialtrans)

        x_spatialtrans = x_spatialtrans

        # plt.subplot(413)
        # plt.imshow(x_spatialtrans[:, 0, :].cpu().detach().numpy(), cmap='gray')

        #sin and cos positional embedding
        #spatial trans 基于step的position embedding
        # x_pos_spatialtrans = x.permute(1,2,0).contiguous()#B,C,X
        # pos_encoder = PositionalEncoding1D(26).to(device='cuda')
        # x_pos_spatialtrans = pos_encoder(x_pos_spatialtrans)
        # x_pos_spatialtrans = x_pos_spatialtrans.permute(2,0,1).contiguous() #(x,b,c) #(25,100,204)
        # print('x_pos_spatialtrans',x_pos_spatialtrans.shape)

        #spatial trans 基于channel的positional embeddimg
        # x_pos_2_spatialtrans = x.permute(1,0,2).contiguous() #B,X,C
        # pos_encoder_2 = PositionalEncoding1D(204).to(device='cuda')
        # x_pos_2_spatialtrans = pos_encoder_2(x_pos_2_spatialtrans)
        # x_pos_2_spatialtrans = x_pos_2_spatialtrans.permute(1,0,2).contiguous() #(x,b,c) (25,100,204)
        # print('0000', x_pos_2_spatialtrans.shape)

        #spatial Linear Projection
        x_spatialtrans = x_spatialtrans.permute(1,0,2).contiguous()
        print('1111',x_spatialtrans.shape)
        x_spatialtrans = self.linearprojection_spatialtrans(x_spatialtrans)
        x_spatialtrans = x_spatialtrans.permute(1,0,2).contiguous()
        x_spatialtrans = x_spatialtrans  #加不加position encoding?  #(25,100,204)

        #cls_token和pos_embedding
        x_spatialtrans = x_spatialtrans.permute(1, 0, 2).contiguous() #(B,X,C)
        b, n, _ = x_spatialtrans.shape
        cls_tokens_spatial = repeat(self.cls_token_spatial, '() n d -> b n d', b=b)
        x_spatialtrans = torch.cat((cls_tokens_spatial, x_spatialtrans), dim=1)
        print('xxx',x_spatialtrans.shape)
        x_spatialtrans += self.pos_embedding_spatial[:, :(x_spatialtrans.size(1) + 1)]
        x_spatialtrans = x_spatialtrans.permute(1, 0, 2).contiguous()  # (X,B,C)
        print('xxx',x_spatialtrans.shape)

        # 设置transformer参数
        encoder_layer_spatial = nn.TransformerEncoderLayer(d_model=204, nhead=1, dim_feedforward=32,activation='gelu').to(device='cuda')
        transformer_encoder_spatial = nn.TransformerEncoder(encoder_layer_spatial, num_layers=1, norm=None).to(device='cuda')

        # 最后训练 spatial transformer
        x_spatialtrans_output = transformer_encoder_spatial(x_spatialtrans) # (25,100,204)
        x_spatialtrans_output = self.dense_layer_spatial(x_spatialtrans_output) #attention之后的全连接层
        x_spatialtrans_output = self.relu(x_spatialtrans_output)
        # plt.subplot(414)
        # plt.imshow(x_spatialtrans_output[:, 0, :].cpu().detach().numpy(), cmap='gray')

        #transformer的output进行变形 再卷积 再transformer
        # x_trans_2 = x_trans.reshape(x_trans.shape[1],x_trans.shape[2],5,5)
        # x_trans_2 = self.conv2d_4(x_trans_2)
        # x_trans_2 = self.relu(x_trans_2) #(B,C,H,W)
        # print('x_tran_2',x_trans_2.shape)
        # x_2 = x_trans_2.reshape(-1,x_trans_2.shape[0], x_trans_2.shape[1])
        # print('1',x_2.shape) #S,B,C
        # x_pos_2 = x_2.permute(1, 0, 2).contiguous()
        # pos_encoder_2 = PositionalEncoding1D(512).to(device='cuda')
        # x_pos_2 = pos_encoder_2(x_pos_2)
        # x_pos_2 = x_pos_2.permute(1, 0, 2).contiguous()
        # print('2',x_pos_2.shape)
        # encoder_layer2 = nn.TransformerEncoderLayer(d_model=512, nhead=1, dim_feedforward=32, activation='relu').to(
        #     device='cuda')
        # transformer_encoder_2 = nn.TransformerEncoder(encoder_layer2, num_layers=2, norm=None).to(device='cuda')
        # x_trans_2 = transformer_encoder_2(x * x_pos)  # (25,100,200)

        x = x_spatialtrans_output #(N,B,C)
        # x_mlp = x.permute(1,0,2).contiguous() #(B,N,C)
        # print('xxx1', x.shape)
        # x_mlp = x_mlp.mean(dim=1) if self.pool == 'mean' else x_mlp[:, 0]
        # print('xxx2', x_mlp.shape)
        # x_mlp = self.to_latent(x_mlp)
        # print('xxx3', x_mlp.shape)

        # plt.subplot(414)
        # plt.imshow(x[:, 0, :].cpu().detach().numpy(), cmap='gray')
        # plt.show()

        #选择中心pixel的值
        x = self.ave1Dpooling(x) #(N,B,C)
        x = self.prelu(x)
        x_center = x[13,:,:] #(B,C)
        x_center = x_center.permute(0, 1).contiguous()
        x = x_center.view(x_center.size(0), -1)
        x = x.view(x.size(0), -1)


        #batchnorm+fc 分类工作
        x = self.gru_bn_trans(x)
        x = self.prelu(x)
        x = self.dropout(x)
        preds_SpaViT = self.fc_trans_1(x)


        preds_joint = torch.cat([preds_SpeViT,preds_SpaViT],dim=1)
        print('preds_joint',preds_joint.shape)
        preds_joint = self.fc_joint(preds_joint)

        # x = self.gru_bn(x)
        # x = self.fc(x)
        # print('preds',x.shape)

        #作图
        # plt.grid(linewidth=0.5, color='black')
        # plt.title('Real situation in one patch', fontdict={'size': 40})
        # plt.xlabel('Band Numbers', fontdict={'size': 40}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 40})
        # plt.xticks(fontsize=35)
        # plt.yticks(fontsize=35)
        # bwith = 2  # 边框宽度设置为2
        # TK = plt.gca()  # 获取边框
        # TK.spines['bottom'].set_linewidth(bwith)  # 图框下边
        # TK.spines['left'].set_linewidth(bwith)  # 图框左边
        # TK.spines['top'].set_linewidth(bwith)  # 图框上边
        # TK.spines['right'].set_linewidth(bwith)  # 图框右边
        # plt.show()
        # x = self.fc2(x)
        return  preds_joint + preds_SpeViT + preds_SpaViT

class zhou3dvit(nn.Module):

    @staticmethod
    def weight_init(m):
        # All weight matrices in our RNN and bias vectors are initialized with a uniform distribution, and the values of these weight matrices and bias vectors are initialized in the range [−0.1,0.1]
        if isinstance(m, (nn.Linear, nn.GRU, nn.Conv3d,nn.LSTM,nn.Conv2d,nn.TransformerEncoder,nn.TransformerEncoderLayer)):
            init.kaiming_uniform_(m.weight.data)
            init.kaiming_uniform_(m.bias.data)

    def __init__(self, input_channels, n_classes, patch_size=9, pool = 'cls', embed_dim=64):
        # The proposed network model uses a single recurrent layer that adopts our modified GRUs of size 64 with sigmoid gate activation and PRetanh activation functions for hidden representations
        super(zhou3dvit, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.gru = nn.GRU(patch_size**2 , patch_size**2, 1, bidirectional=False, batch_first=False)  # TODO: try to change this ?
        self.conv1d = nn.Conv1d(in_channels=25, out_channels=25, kernel_size=3, stride=1)
        self.conv2d_1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=2, stride=1)
        self.conv2d_2 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=2, stride=1)
        self.conv2d_3 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=2, stride=1)
        self.conv2d_4 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1)
        self.conv2d_5_1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1)
        self.conv2d_5_2 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1)
        self.conv2d_5_3 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1)
        self.projection_3d = nn.Linear(90,embed_dim)
        self.lstm_2 = nn.LSTM(input_channels, input_channels, 1, bidirectional=False)
        self.gru_bn = nn.BatchNorm1d(patch_size**2 * (input_channels))
        self.gru_bn_trans = nn.BatchNorm1d(input_channels)
        self.gru_bn_2 = nn.BatchNorm1d((patch_size ** 2) * 64)
        self.gru_bn_spectralclasstoken = nn.BatchNorm1d(embed_dim)
        self.gru_bn_spatialclasstoken = nn.BatchNorm1d(embed_dim)
        self.pos_embedding_spectral = nn.Parameter(torch.randn(1, embed_dim + 1, embed_dim))
        self.pos_embedding_spatial = nn.Parameter(torch.randn(1, (311) + 1, embed_dim))
        # self.pos_embedding_3d = get_pos_encode2(num_patches_3D, dim, num_h, num_w, num_f)
        # self.pos_embedding_3d = nn.Parameter(torch.randn(1, 12, 272, 3,3))
        self.cls_token_spectral = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.cls_token_spatial = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.elu = nn.ELU()
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        self.ave1Dpooling_spatial = nn.AdaptiveAvgPool1d(1)
        self.ave1Dpooling_spectral = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(patch_size**2 * (input_channels), n_classes)
        self.fc_trans_spatial = nn.Linear(embed_dim, n_classes)
        self.fc_trans_spectral = nn.Linear(embed_dim, n_classes)
        self.dense_layer_spectral = nn.Linear(embed_dim, embed_dim)
        self.dense_layer_spatial = nn.Linear(embed_dim,embed_dim)
        self.fc_2 = nn.Linear((patch_size ** 2) * 64, n_classes)
        self.linearprojection_spatialtrans = nn.Linear(in_features=embed_dim,out_features=embed_dim)
        self.linearprojection_spectraltrans = nn.Linear(in_features=311, out_features=embed_dim)
        self.linearprojection_25_to_81 = nn.Linear(in_features=25, out_features=81)
        self.fc_3 = nn.Linear(64, n_classes)
        self.fc_vit = nn.Linear(input_channels,n_classes)
        self.fc_joint = nn.Linear(n_classes*2,n_classes)
        self.softmax = nn.Softmax()
        self.pool = pool
        self.layernorm = nn.LayerNorm(embed_dim)
        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b (c c1) (h h1) (w w1) -> b () (h1 w1 c1)', p1 = 2, p2 = 2))
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


    def forward(self, x): #初始是第1方向
        print('x.shape', x.shape)

        x = x.squeeze(1)  # B,F,P,P

        # ResNet patch_size = 9 for SA PU
        # x = self.conv2d_1(x)
        # print('1', x.shape)
        # x = self.relu(x)
        # x = self.conv2d_2(x)
        # print('2', x.shape)
        # x_res = self.relu(x)
        # x_res = self.conv2d_3(x_res)
        # print('3', x.shape) #(ptach size = 6)
        # x_res = self.relu(x_res)
        # x_res_res = self.conv2d_4(x_res)
        # x_res_res = self.relu(x_res_res)
        # x = x_res + x_res_res
        # print('4', x.shape) #SA(b,204,6,6)

        #直接用3DCNN试试
        # x = repeat(x, 'b d h w -> b () d h w')
        # threedcnn_1 = nn.Conv3d(in_channels=1,out_channels=self.embed_dim,kernel_size=3,stride=1).to(device='cuda')
        # y = threedcnn_1(x)
        # print('y',y.shape)#(b, emded__dim, d,4,4)
        # threedcnn_2 = nn.Conv3d(in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=3, stride=1).to(device='cuda')
        # y = threedcnn_2(y)
        # y = reduce(y,'b c d h w -> b c d 1 1',reduction='mean')
        # y = reduce(y, 'b c d 1 1 -> b c d', reduction='mean')
        # print('y',y.shape)


        #try the 3D overlapped slice
        patch_h = 3
        patch_w = 3
        patch_c = 10
        y = patch_3D(x, kernel_size=(patch_h,patch_w,patch_c),padding=(0,0,0),stride=(3,3,10))
        print('overlapped3d_x',y.shape)
        y_3d = rearrange(y, 'b num (h w c) -> b c num h w',c = patch_c, h=patch_h, w=patch_w) #暂时没用到
        # print('overlapped3d_x', y.shape)
        # y = y + self.pos_embedding_3d
        y = self.projection_3d(y)
        print('overlapped3d_x', y.shape)
        pe_h,pe_w,pe_f = get_pos_encode2(num_patches_3D=180, dim=64, num_h=3, num_w=3, num_f=10)
        print('pe_h',pe_h.shape)
        # y = reduce(y, 'b c d h w -> b (c h w) d', reduction='mean')
        print('overlapped3d_x', y.shape) #SA(b, embed-dim, 3213) 把3213看成步长, embde_dim看成channel所以就是(b,c,x)


        # try the non-overllaped slice
        # y = rearrange(x, 'b c h w -> b h w c')
        # y = rearrange(y, 'b (h patch_height) (w patch_width) (c patch_channel) -> b (h w c) (patch_height patch_width patch_channel)'
        #               , patch_height=2, patch_width=2, patch_channel=4)
        # y = rearrange(y, 'b num (h1 w1 c1) -> b num h1 w1 c1', h1=2, w1 =2, c1=4) #(b,d,h,w,c) SA:(b,612,3,3,3)
        # print('rearrange', y.shape)
        # y = rearrange(y, 'b d h w c -> b c d h w') #SA(b,3,612,3,3)
        # print('rearrange', y.shape)
        # y = self.projection_3d(y)
        # print('rearrange', y.shape) #(b,c,d h w) SA(b, embed_dim, 204, 1 , 1)
        # y = reduce(y, 'b c d h w -> b (c h w) d', reduction='mean') #SA(b, embed-dim, 204) 把204看成步长, embde_dim看成channel所以就是(b,c,x)
        # print('rearrange', y.shape) #(btach, embed_dim, 204)


        # ResNet patch_size = 5 for IP dataset
        # x_spectraltrans_reconstruct = self.conv2d_5_1(x)
        # x_spectraltrans_reconstruct_1 = x_spectraltrans_reconstruct
        # x_spectraltrans_reconstruct_2 = self.relu(x_spectraltrans_reconstruct)
        # x_spectraltrans_reconstruct_2 = self.conv2d_5_2(x_spectraltrans_reconstruct_2)
        # x_spectraltrans_reconstruct_2 = self.relu(x_spectraltrans_reconstruct_2)
        # x_spectraltrans_reconstruct_3 = self.conv2d_5_3(x_spectraltrans_reconstruct_2)
        # x_spectraltrans_reconstruct_3 = self.relu(x_spectraltrans_reconstruct_2)
        # x = x_spectraltrans_reconstruct_1 + x_spectraltrans_reconstruct_2 + x_spectraltrans_reconstruct_3


        # Transformerencoder 从这里开始
        # 3d spectral trans (SpeViT)
        x_spectraltrans = y.permute(1, 0, 2).contiguous()  # (C,B,X)
        x_spectraltrans = x_spectraltrans.permute(1, 0, 2).contiguous()  # (B,C,X) SA(100, 128, 204)

        # spectral Linear Projection
        x_spectraltrans = self.linearprojection_spectraltrans(x_spectraltrans)
        print('spectral_linearpro1', x_spectraltrans.shape) #SA(100, 128, 25)
        x_spectraltrans = x_spectraltrans.permute(1, 0, 2).contiguous()  # (C,B,X)

        # spectral cls_token和pos_embedding
        x_spectraltrans = x_spectraltrans.permute(1, 0, 2).contiguous()  # (B,C,X)
        print('spectral_linearpro2', x_spectraltrans.shape) #SA(100, 128, 25)
        b_spectral, n_spectral, _ = x_spectraltrans.shape
        cls_tokens_spectral = repeat(self.cls_token_spectral, '() n_spectral d_spectral -> b_spectral n_spectral d_spectral', b_spectral=b_spectral)
        x_spectraltrans = torch.cat((cls_tokens_spectral, x_spectraltrans), dim=1)
        print('spectral_linearpro3', x_spectraltrans.shape) #SA(100, 129, 25)
        x_spectraltrans = x_spectraltrans + self.pos_embedding_spectral[:, :(x_spectraltrans.size(1) + 1)]
        print('spectral_linearpro4', x_spectraltrans.shape) #SA(100, 129, 25)
        x_spectraltrans = x_spectraltrans.permute(1, 0, 2).contiguous()  # (C,B,X)
        print('spectral_linearpro5', x_spectraltrans.shape) #SA(129, 100, 25)

        #设置spectral transformer参数
        encoder_layer_spectral = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=1, dim_feedforward=self.embed_dim,
                                                            activation='gelu').to(device='cuda')
        transformer_encoder_spectral = nn.TransformerEncoder(encoder_layer_spectral, num_layers=1, norm=None).to(
            device='cuda')
        #最后训练 spectral transformer
        x_spectraltrans_output = transformer_encoder_spectral(x_spectraltrans)
        print('spectral_linearpro6', x_spectraltrans.shape) #SA(129, 100, 25) (L,N,C)
        x_spectraltrans_output = self.layernorm(x_spectraltrans_output)
        x_spectraltrans_output = self.dense_layer_spectral(x_spectraltrans_output)  # attention之后的全连接层 #SA(129, 100, 25) (L,N,C)

        x_spectraltrans_output = self.relu(x_spectraltrans_output) #SA(129, 100, 25) (L_in,N,C)
        #
        # SpeViT的output layer
        x_spectraltrans_output = rearrange(x_spectraltrans_output, 'length_in batch channel -> batch channel length_in')  #(N,C,L_in)

        x_spectral_classtoken = reduce(x_spectraltrans_output, 'batch channel length_in -> batch channel 1', reduction='mean')

        # x_spectral_classtoken = self.ave1Dpooling_spectral(x_spectraltrans_output)

        x_spectral_classtoken = self.relu(x_spectral_classtoken)

        print('spectral_linearpro7', x_spectral_classtoken.shape) #(batch, channel, length)

        x_spectral_classtoken = x_spectral_classtoken.view(x_spectral_classtoken.size(0), -1)
        x_spectral_classtoken = self.gru_bn_spectralclasstoken(x_spectral_classtoken)
        # x_spectral_classtoken = self.prelu(x_spectral_classtoken)
        x_spectral_classtoken = self.dropout(x_spectral_classtoken)
        preds_SpeViT = self.fc_trans_spectral(x_spectral_classtoken)  # SpeViT de output

        #------------------------------------------------------------------分割线----------------------------------------------------------------------------------------------#
        # 进入3d SpaViT
        x_spatialtrans = y.permute(2,0,1)  # (x,b,c) SA(204, 100, embded_dim)
        print('spatial_linearpro1', x_spatialtrans.shape) #

        # spatial Linear Projection
        x_spatialtrans = x_spatialtrans.permute(1, 0, 2).contiguous()
        print('spatial_linearpro2', x_spatialtrans.shape) #SA(100, 204, embded_dim) (b,x,c)
        x_spatialtrans = self.linearprojection_spatialtrans(x_spatialtrans)
        print('spatial_linearpro3', x_spatialtrans.shape) #SA(100, 204, 204) (b,x,c)
        x_spatialtrans = x_spatialtrans.permute(1, 0, 2).contiguous()
        x_spatialtrans = x_spatialtrans
        print('spatial_linearpro4', x_spatialtrans.shape) #SA(204,100,204) (x,b,c)

        #spatial cls_token和pos_embedding
        x_spatialtrans = x_spatialtrans.permute(1, 0, 2).contiguous()  # (B,X,C)
        b_spatial, n_spatial, _ = x_spatialtrans.shape
        cls_tokens_spatial = repeat(self.cls_token_spatial, '() n_spatial d_spatial -> b_spatial n_spatial d_spatial', b_spatial=b_spatial)
        x_spatialtrans = torch.cat((cls_tokens_spatial, x_spatialtrans), dim=1)
        print('spatial_linearpro5', x_spatialtrans.shape)#SA(b, x+1,c) (100, 205, 204)
        x_spatialtrans = x_spatialtrans + self.pos_embedding_spatial[:, :(x_spatialtrans.size(1) + 1)]
        print('spatial_linearpro6', x_spatialtrans.shape) #SA(b, x+1,c) (100, 205, 204)
        x_spatialtrans = x_spatialtrans.permute(1, 0, 2).contiguous()  # (X,B,C) SA(205,100,204

        # 设置spatial transformer参数
        encoder_layer_spatial = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=1, dim_feedforward=self.embed_dim,
                                                           activation='gelu').to(device='cuda')
        transformer_encoder_spatial = nn.TransformerEncoder(encoder_layer_spatial, num_layers=1, norm=None).to(
            device='cuda')

        # 最后训练 spatial transformer
        x_spatialtrans_output = transformer_encoder_spatial(x_spatialtrans)  # SA(205,100,204) (x+1, b , c)
        x_spatialtrans_output = self.layernorm(x_spatialtrans_output)

        x_spatialtrans_output = self.dense_layer_spatial(x_spatialtrans_output)  # attention之后的全连接层 # SA(205,100,204) (x+1, b , c)

        x_spatialtrans_output = self.relu(x_spatialtrans_output) #SA(x+1,b,c) (length_in, n, c)

        # 3d SpaViT的output layer
        x_spatialtrans_output = rearrange(x_spatialtrans_output, 'length_in batch channel -> batch channel length_in')  # (N,C,L_in)


        # x_spatial_classtoken = self.ave1Dpooling_spatial(x_spatialtrans_output) #(N, C, 1)
        x_spatial_classtoken = reduce(x_spatialtrans_output, 'batch channel length_in -> batch channel 1', reduction='mean')

        print('spatial_linearpro7', x_spatial_classtoken.shape)

        x_spatial_classtoken = self.relu(x_spatial_classtoken)

        print('spatial_linearpro8', x_spatial_classtoken.shape)

        x_spatial_classtoken = x_spatial_classtoken.view(x_spatial_classtoken.size(0), -1)
        print('spatial_linearpro9', x_spatial_classtoken.shape)
        x_spatial_classtoken = self.gru_bn_spatialclasstoken(x_spatial_classtoken)
        x_spatial_classtoken = self.dropout(x_spatial_classtoken)
        preds_SpaViT = self.fc_trans_spatial(x_spatial_classtoken)  # SpaViT de output

        # plt.subplot(414)
        # plt.imshow(x_spatialtrans_output[:, 0, :].cpu().detach().numpy(), cmap='gray')

        # x = x_spatialtrans_output  # (N,B,C)
        # print('output', x.shape)
        #
        # # 选择中心pixel的值
        # x = self.ave1Dpooling_spatial(x)  # (N,B,C)
        # print('output',x.shape)
        # x = self.prelu(x)
        # x_center = x[5, :, :]  # (B,C)
        # x_center = x_center.permute(0, 1).contiguous()
        # x = x_center.view(x_center.size(0), -1)
        # x = x.view(x.size(0), -1)
        #
        # # batchnorm+fc 分类工作
        # x = self.gru_bn_trans(x)
        # x = self.prelu(x)
        # x = self.dropout(x)
        # preds_SpaViT = self.fc_trans_spatial(x)



        # x = self.gru_bn(x)
        # x = self.fc(x)
        # print('preds',x.shape)


        return  preds_SpaViT

class zhouICIP2022(nn.Module):

    @staticmethod
    def weight_init(m):
        # All weight matrices in our RNN and bias vectors are initialized with a uniform distribution, and the values of these weight matrices and bias vectors are initialized in the range [−0.1,0.1]
        if isinstance(m, (nn.Linear, nn.GRU, nn.Conv3d,nn.LSTM,nn.Conv2d,nn.TransformerEncoder,nn.TransformerEncoderLayer)):
            init.kaiming_uniform_(m.weight.data)
            init.kaiming_uniform_(m.bias.data)

    def __init__(self, input_channels, n_classes, patch_size=9, pool = 'cls', embed_dim=64):
        # The proposed network model uses a single recurrent layer that adopts our modified GRUs of size 64 with sigmoid gate activation and PRetanh activation functions for hidden representations
        super(zhouICIP2022, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.gru = nn.GRU(patch_size**2 , patch_size**2, 1, bidirectional=False, batch_first=False)  # TODO: try to change this ?
        self.conv1d = nn.Conv1d(in_channels=25, out_channels=25, kernel_size=3, stride=1)
        self.conv2d_1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=2, stride=1)
        self.conv2d_2 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=2, stride=1)
        self.conv2d_3 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=2, stride=1)
        self.conv2d_4 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1)
        self.conv2d_5_1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1)
        self.conv2d_5_2 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1)
        self.conv2d_5_3 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1)
        self.projection_1d = nn.Linear(90,embed_dim)
        self.projection_3d = nn.Conv3d(in_channels=10, out_channels=16, kernel_size=(1,2,2),stride=(1, 1, 1) )
        self.lstm_2 = nn.LSTM(input_channels, input_channels, 1, bidirectional=False)
        self.gru_bn = nn.BatchNorm1d(patch_size**2 * (input_channels))
        self.gru_bn_trans = nn.BatchNorm1d(input_channels)
        self.gru_bn_2 = nn.BatchNorm1d((patch_size ** 2) * 64)
        self.gru_bn_spectralclasstoken = nn.BatchNorm1d(180)
        self.gru_bn_spatialclasstoken = nn.BatchNorm1d(embed_dim)
        self.pos_embedding_spectral = nn.Parameter(torch.randn(1, embed_dim + 1, 180))
        self.pos_embedding_spatial = nn.Parameter(torch.randn(1, (180) + 1, embed_dim))
        self.pos_embedding_3d = get_pos_encode2(180, embed_dim, 3, 3, 20)
        # self.pos_embedding_3d = nn.Parameter(torch.randn(1, 12, 272, 3,3))
        self.cls_token_spectral = nn.Parameter(torch.randn(1, 1, 180))
        self.cls_token_spatial = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.elu = nn.ELU()
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        self.ave1Dpooling_spatial = nn.AdaptiveAvgPool1d(1)
        self.ave1Dpooling_spectral = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(patch_size**2 * (input_channels), n_classes)
        self.fc_trans_spatial = nn.Linear(embed_dim, n_classes)
        self.fc_trans_spectral = nn.Linear(180, n_classes)
        self.dense_layer_spectral = nn.Linear(180, 180)
        self.dense_layer_spatial = nn.Linear(embed_dim,embed_dim)
        self.fc_2 = nn.Linear((patch_size ** 2) * 64, n_classes)
        self.linearprojection_spatialtrans = nn.Linear(in_features=embed_dim,out_features=embed_dim)
        self.linearprojection_spectraltrans = nn.Linear(in_features=180, out_features=180)
        self.linearprojection_25_to_81 = nn.Linear(in_features=25, out_features=81)
        self.linearprojection_3d = nn.Linear(in_features=90, out_features=embed_dim)
        self.fc_3 = nn.Linear(64, n_classes)
        self.fc_vit = nn.Linear(input_channels,n_classes)
        self.fc_joint = nn.Linear(n_classes*2,n_classes)
        self.softmax = nn.Softmax()
        self.pool = pool
        self.layernorm_spe = nn.LayerNorm(180)
        self.layernorm_spa = nn.LayerNorm(embed_dim)
        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b (c c1) (h h1) (w w1) -> b () (h1 w1 c1)', p1 = 2, p2 = 2))
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


    def forward(self, x): #初始是第1方向
        print('x.shape', x.shape)

        x = x.squeeze(1)  # B,F,P,P

        # ResNet patch_size = 9 for SA PU
        # x = self.conv2d_1(x)
        # print('1', x.shape)
        # x = self.relu(x)
        # x = self.conv2d_2(x)
        # print('2', x.shape)
        # x_res = self.relu(x)
        # x_res = self.conv2d_3(x_res)
        # print('3', x.shape) #(ptach size = 6)
        # x_res = self.relu(x_res)
        # x_res_res = self.conv2d_4(x_res)
        # x_res_res = self.relu(x_res_res)
        # x = x_res + x_res_res
        # print('4', x.shape) #SA(b,204,6,6)

        #直接用3DCNN试试
        # x = repeat(x, 'b d h w -> b () d h w')
        # threedcnn_1 = nn.Conv3d(in_channels=1,out_channels=self.embed_dim,kernel_size=3,stride=1).to(device='cuda')
        # y = threedcnn_1(x)
        # print('y',y.shape)#(b, emded__dim, d,4,4)
        # threedcnn_2 = nn.Conv3d(in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=3, stride=1).to(device='cuda')
        # y = threedcnn_2(y)
        # y = reduce(y,'b c d h w -> b c d 1 1',reduction='mean')
        # y = reduce(y, 'b c d 1 1 -> b c d', reduction='mean')
        # print('y',y.shape)


        #try the 3D overlapped slice
        patch_h = 3
        patch_w = 3
        patch_c = 10
        y = patch_3D(x, kernel_size=(patch_h,patch_w,patch_c),padding=(0,0,0),stride=(3,3,10))
        print('overlapped3d_x',y.shape)
        y_3d = rearrange(y, 'b num (h w c) -> b c num h w',c = patch_c, h=patch_h, w=patch_w) #暂时没用到 (100 10 180 3 3)
        print('y_3d', y_3d.shape)

        # y = y + self.pos_embedding_3d
        y_1d = self.projection_1d(y)
        print('y', y.shape)  # (100, 180, 64)
        y_3d = self.projection_3d(y_3d)
        print('y_3d', y_3d.shape)
        y_3d = y_3d.view(y_3d.size(0),y_3d.size(2),-1)
        # y_3d = self.linearprojection_3d(y_3d)
        print('y_3d', y_3d.shape)

        pe_h,pe_w,pe_f = self.pos_embedding_3d
        y_1d = y_1d + pe_h[:,1:181,:].to(device='cuda') + pe_w[:,1:181,:].to(device='cuda') + pe_f[:,1:181,:].to(device='cuda')
        print('pe_h',pe_h.shape)
        # y = reduce(y, 'b c d h w -> b (c h w) d', reduction='mean')
        # print('overlapped3d_x', y.shape) #SA(b, embed-dim, 3213) 把3213看成步长, embde_dim看成channel所以就是(b,c,x) (100,180,64)


        # try the non-overllaped slice
        # y = rearrange(x, 'b c h w -> b h w c')
        # y = rearrange(y, 'b (h patch_height) (w patch_width) (c patch_channel) -> b (h w c) (patch_height patch_width patch_channel)'
        #               , patch_height=2, patch_width=2, patch_channel=4)
        # y = rearrange(y, 'b num (h1 w1 c1) -> b num h1 w1 c1', h1=2, w1 =2, c1=4) #(b,d,h,w,c) SA:(b,612,3,3,3)
        # print('rearrange', y.shape)
        # y = rearrange(y, 'b d h w c -> b c d h w') #SA(b,3,612,3,3)
        # print('rearrange', y.shape)
        # y = self.projection_3d(y)
        # print('rearrange', y.shape) #(b,c,d h w) SA(b, embed_dim, 204, 1 , 1)
        # y = reduce(y, 'b c d h w -> b (c h w) d', reduction='mean') #SA(b, embed-dim, 204) 把204看成步长, embde_dim看成channel所以就是(b,c,x)
        # print('rearrange', y.shape) #(btach, embed_dim, 204)


        # ResNet patch_size = 5 for IP dataset
        # x_spectraltrans_reconstruct = self.conv2d_5_1(x)
        # x_spectraltrans_reconstruct_1 = x_spectraltrans_reconstruct
        # x_spectraltrans_reconstruct_2 = self.relu(x_spectraltrans_reconstruct)
        # x_spectraltrans_reconstruct_2 = self.conv2d_5_2(x_spectraltrans_reconstruct_2)
        # x_spectraltrans_reconstruct_2 = self.relu(x_spectraltrans_reconstruct_2)
        # x_spectraltrans_reconstruct_3 = self.conv2d_5_3(x_spectraltrans_reconstruct_2)
        # x_spectraltrans_reconstruct_3 = self.relu(x_spectraltrans_reconstruct_2)
        # x = x_spectraltrans_reconstruct_1 + x_spectraltrans_reconstruct_2 + x_spectraltrans_reconstruct_3


        # Transformerencoder 从这里开始
        # 3d spectral trans (SpeViT)
        x_spectraltrans = y_1d.permute(0, 2, 1).contiguous()  # (C,B,X)(100,64,180) 把64当做步长

        print('x_spectraltrans', x_spectraltrans.shape)

        # spectral Linear Projection
        x_spectraltrans = self.linearprojection_spectraltrans(x_spectraltrans)
        print('spectral_linearpro1', x_spectraltrans.shape) #SA(100, c64, x180)
        x_spectraltrans = x_spectraltrans.permute(1, 0, 2).contiguous()  # (C64,B100,X180)

        # spectral cls_token和pos_embedding
        x_spectraltrans = x_spectraltrans.permute(1, 0, 2).contiguous()  # (B100,C64,X180)
        print('spectral_linearpro2', x_spectraltrans.shape) #SA(100, c64, x180)
        b_spectral, n_spectral, _ = x_spectraltrans.shape
        cls_tokens_spectral = repeat(self.cls_token_spectral, '() n_spectral d_spectral -> b_spectral n_spectral d_spectral', b_spectral=b_spectral)
        x_spectraltrans = torch.cat((cls_tokens_spectral, x_spectraltrans), dim=1)
        print('spectral_linearpro3', x_spectraltrans.shape) #SA(100, c65, x180)
        x_spectraltrans = x_spectraltrans + self.pos_embedding_spectral[:, :(x_spectraltrans.size(1) + 1)]
        print('spectral_linearpro4', x_spectraltrans.shape) #SA(100, c65, x180)
        x_spectraltrans = x_spectraltrans.permute(1, 0, 2).contiguous()  # (C65,B100,X180)
        print('spectral_linearpro5', x_spectraltrans.shape) #SA(c65, b100, x180)

        #设置spectral transformer参数
        encoder_layer_spectral = nn.TransformerEncoderLayer(d_model=180, nhead=1, dim_feedforward=self.embed_dim,
                                                            activation='gelu').to(device='cuda')
        transformer_encoder_spectral = nn.TransformerEncoder(encoder_layer_spectral, num_layers=1, norm=None).to(
            device='cuda')
        #最后训练 spectral transformer
        x_spectraltrans_output = transformer_encoder_spectral(x_spectraltrans)
        print('spectral_linearpro6', x_spectraltrans.shape) #SA(c65, b100, x180) (L,N,C)
        x_spectraltrans_output = self.layernorm_spe(x_spectraltrans_output)
        x_spectraltrans_output = self.dense_layer_spectral(x_spectraltrans_output)  # attention之后的全连接层 #SA(c65, b100, x180) (L,N,C)

        x_spectraltrans_output = self.relu(x_spectraltrans_output) #SA(c65, b100, x180) (L_in,N,C)
        #
        # SpeViT的output layer
        x_spectraltrans_output = rearrange(x_spectraltrans_output, 'length_in batch channel -> batch channel length_in')  #(N,C,L_in)

        x_spectral_classtoken = reduce(x_spectraltrans_output, 'batch channel length_in -> batch channel 1', reduction='mean')

        # x_spectral_classtoken = self.ave1Dpooling_spectral(x_spectraltrans_output)

        x_spectral_classtoken = self.relu(x_spectral_classtoken)

        print('spectral_linearpro7', x_spectral_classtoken.shape) #(batch, channel, length)

        x_spectral_classtoken = x_spectral_classtoken.view(x_spectral_classtoken.size(0), -1)
        x_spectral_classtoken = self.gru_bn_spectralclasstoken(x_spectral_classtoken)
        # x_spectral_classtoken = self.prelu(x_spectral_classtoken)
        x_spectral_classtoken = self.dropout(x_spectral_classtoken)
        preds_SpeViT = self.fc_trans_spectral(x_spectral_classtoken)  # SpeViT de output#(SpeViT)

        #------------------------------------------------------------------分割线----------------------------------------------------------------------------------------------#
        # # 进入3d SpaViT
        x_spatialtrans = y_1d  # (x,b,c) SA(100, 180, 64)

        # spatial Linear Projection
        # x_spatialtrans = x_spatialtrans.permute(1, 0, 2).contiguous()
        # print('spatial_linearpro2', x_spatialtrans.shape) #SA(100, 204, embded_dim) (b,x,c)
        x_spatialtrans = self.linearprojection_spatialtrans(x_spatialtrans) #(100,180,64)
        # print('spatial_linearpro3', x_spatialtrans.shape) #SA(100, 204, 204) (b,x,c)
        # x_spatialtrans = x_spatialtrans.permute(1, 0, 2).contiguous()
        x_spatialtrans = x_spatialtrans


        #spatial cls_token和pos_embedding
        # x_spatialtrans = x_spatialtrans.permute(1, 0, 2).contiguous()  # (B,X,C)
        b_spatial, n_spatial, _ = x_spatialtrans.shape
        cls_tokens_spatial = repeat(self.cls_token_spatial, '() n_spatial d_spatial -> b_spatial n_spatial d_spatial', b_spatial=b_spatial)
        x_spatialtrans = torch.cat((cls_tokens_spatial, x_spatialtrans), dim=1)

        print('spatial_linearpro5', x_spatialtrans.shape)#SA(100, 181,64)

        # x_spatialtrans += pe_h.to(device='cuda')
        # x_spatialtrans += pe_w.to(device='cuda')
        # x_spatialtrans += pe_f.to(device='cuda')

        # plt.subplot(151)
        # plt.imshow(pe_h[0, :, :].cpu().detach().numpy(),cmap='Greens')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Pos.embedding in height', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('feature dimension', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('number of tokens', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=15)
        #
        # plt.subplot(152)
        # plt.imshow(pe_w[0, :, :].cpu().detach().numpy(),cmap='Greens')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Pos.embedding in width', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('feature dimension', fontdict={'size': 20}, fontweight='bold')
        # # plt.ylabel('number of tokens', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=15)
        #
        # plt.subplot(153)
        # plt.imshow(pe_f[0, :, :].cpu().detach().numpy(),cmap='Greens')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Pos.embedding in dimension', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('feature dimension', fontdict={'size': 20}, fontweight='bold')
        # # plt.ylabel('number of tokens', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=15)
        #
        # plt.subplot(154)
        # plt.imshow((pe_f + pe_h + pe_w)[0, :, :].cpu().detach().numpy(),cmap='Greens')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('sum of pos.embedding', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('feature dimension', fontdict={'size': 20}, fontweight='bold')
        # # plt.ylabel('number of tokens', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=15)


        x_spatialtrans += self.pos_embedding_spatial[:, :(x_spatialtrans.size(1) + 1)]
        # plt.subplot(155)
        # plt.imshow(self.pos_embedding_spatial[0, :, :].cpu().detach().numpy(),cmap='Greens')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Learned pos.embedding', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('feature dimension', fontdict={'size': 20}, fontweight='bold')
        # # plt.ylabel('number of tokens', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=15)



        # plt.show()
        print('spatial_linearpro6', x_spatialtrans.shape)#(100,181,64)
        x_spatialtrans = x_spatialtrans.permute(1, 0, 2).contiguous()  # (x181,100,c64)

        # 设置spatial transformer参数
        encoder_layer_spatial = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=1, dim_feedforward=self.embed_dim,
                                                           activation='gelu').to(device='cuda')
        transformer_encoder_spatial = nn.TransformerEncoder(encoder_layer_spatial, num_layers=1, norm=None).to(
            device='cuda')

        # 最后训练 spatial transformer
        x_spatialtrans_output = transformer_encoder_spatial(x_spatialtrans)  # SA(205,100,204) (x+1, b , c)
        x_spatialtrans_output = self.layernorm_spa(x_spatialtrans_output)

        x_spatialtrans_output = self.dense_layer_spatial(x_spatialtrans_output)  # attention之后的全连接层 # SA(205,100,204) (x+1, b , c)

        x_spatialtrans_output = self.relu(x_spatialtrans_output) #SA(x+1,b,c) (length_in, n, c)

        # 3d SpaViT的output layer
        x_spatialtrans_output = rearrange(x_spatialtrans_output, 'length_in batch channel -> batch channel length_in')  # (N,C,L_in)


        # x_spatial_classtoken = self.ave1Dpooling_spatial(x_spatialtrans_output) #(N, C, 1)
        x_spatial_classtoken = reduce(x_spatialtrans_output, 'batch channel length_in -> batch channel 1', reduction='mean')

        print('spatial_linearpro7', x_spatial_classtoken.shape)

        x_spatial_classtoken = self.relu(x_spatial_classtoken)

        print('spatial_linearpro8', x_spatial_classtoken.shape)

        x_spatial_classtoken = x_spatial_classtoken.view(x_spatial_classtoken.size(0), -1)
        print('spatial_linearpro9', x_spatial_classtoken.shape)
        x_spatial_classtoken = self.gru_bn_spatialclasstoken(x_spatial_classtoken)
        x_spatial_classtoken = self.dropout(x_spatial_classtoken)
        preds_SpaViT = self.fc_trans_spatial(x_spatial_classtoken)  # SpaViT de output



        return  preds_SpaViT

class zhouEightDRNN(nn.Module):

    @staticmethod
    def weight_init(m):
        # All weight matrices in our RNN and bias vectors are initialized with a uniform distribution, and the values of these weight matrices and bias vectors are initialized in the range [−0.1,0.1]
        if isinstance(m, (nn.Linear, nn.GRU, nn.Conv3d, nn.LSTM, nn.GRU, nn.Parameter)):
            init.kaiming_uniform_(m.weight.data)
            init.kaiming_uniform_(m.bias.data)

    def __init__(self, input_channels, n_classes, patch_size=5, emb_size = 64):
        # The proposed network model uses a single recurrent layer that adopts our modified GRUs of size 64 with sigmoid gate activation and PRetanh activation functions for hidden representations
        super(zhouEightDRNN, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.gru = nn.GRU(patch_size ** 2, patch_size ** 2, 1, bidirectional=False,
                          batch_first=False)  # TODO: try to change this ?
        self.gru_2 = nn.LSTM(input_channels, input_channels, 1, bidirectional=False)
        self.gru_3 = nn.LSTM(input_channels, patch_size ** 2, 1, bidirectional=True)
        self.pre_emd = nn.Linear(input_channels, emb_size)
        # self.gru_3_1 = nn.LSTM(emb_size, emb_size, 1, bidirectional=False)
        # self.gru_3_2 = nn.LSTM(emb_size, emb_size, 1, bidirectional=False)
        # self.gru_3_3 = nn.LSTM(emb_size, emb_size, 1, bidirectional=False)
        # self.gru_3_4 = nn.LSTM(emb_size, emb_size, 1, bidirectional=False)
        # self.gru_3_5 = nn.LSTM(emb_size, emb_size, 1, bidirectional=False)
        # self.gru_3_6 = nn.LSTM(emb_size, emb_size, 1, bidirectional=False)
        # self.gru_3_7 = nn.LSTM(emb_size, emb_size, 1, bidirectional=False)
        # self.gru_3_8 = nn.LSTM(emb_size, emb_size, 1, bidirectional=False)
        # self.gru_3_1 = nn.LSTM(input_channels, input_channels, 1, bidirectional=False)
        # self.gru_3_2 = nn.LSTM(input_channels, input_channels, 1, bidirectional=False)
        # self.gru_3_3 = nn.LSTM(input_channels, input_channels, 1, bidirectional=False)
        # self.gru_3_4 = nn.LSTM(input_channels, input_channels, 1, bidirectional=False)
        # self.gru_3_5 = nn.LSTM(input_channels, input_channels, 1, bidirectional=False)
        # self.gru_3_6 = nn.LSTM(input_channels, input_channels, 1, bidirectional=False)
        # self.gru_3_7 = nn.LSTM(input_channels, input_channels, 1, bidirectional=False)
        # self.gru_3_8 = nn.LSTM(input_channels, input_channels, 1, bidirectional=False)
        # self.gru_3_1 = nn.RNN(emb_size, emb_size, 1, bidirectional=False)
        # self.gru_3_2 = nn.RNN(emb_size, emb_size, 1, bidirectional=False)
        # self.gru_3_3 = nn.RNN(emb_size, emb_size, 1, bidirectional=False)
        # self.gru_3_4 = nn.RNN(emb_size, emb_size, 1, bidirectional=False)
        # self.gru_3_5 = nn.RNN(emb_size, emb_size, 1, bidirectional=False)
        # self.gru_3_6 = nn.RNN(emb_size, emb_size, 1, bidirectional=False)
        # self.gru_3_7 = nn.RNN(emb_size, emb_size, 1, bidirectional=False)
        # self.gru_3_8 = nn.RNN(emb_size, emb_size, 1, bidirectional=False)
        self.gru_3_1 = nn.GRU(emb_size, emb_size, 1, bidirectional=False)
        self.gru_3_2 = nn.GRU(emb_size, emb_size, 1, bidirectional=False)
        self.gru_3_3 = nn.GRU(emb_size, emb_size, 1, bidirectional=False)
        self.gru_3_4 = nn.GRU(emb_size, emb_size, 1, bidirectional=False)
        self.gru_3_5 = nn.GRU(emb_size, emb_size, 1, bidirectional=False)
        self.gru_3_6 = nn.GRU(emb_size, emb_size, 1, bidirectional=False)
        self.gru_3_7 = nn.GRU(emb_size, emb_size, 1, bidirectional=False)
        self.gru_3_8 = nn.GRU(emb_size, emb_size, 1, bidirectional=False)
        self.alpha_1 = nn.Parameter(torch.randn(1, 1))
        self.alpha_2 = nn.Parameter(torch.randn(1, 1))
        self.alpha_3 = nn.Parameter(torch.randn(1, 1))
        self.alpha_4 = nn.Parameter(torch.randn(1, 1))
        self.alpha_5 = nn.Parameter(torch.randn(1, 1))
        self.alpha_6 = nn.Parameter(torch.randn(1, 1))
        self.alpha_7 = nn.Parameter(torch.randn(1, 1))
        self.alpha_8 = nn.Parameter(torch.randn(1, 1))
        self.gru_bn = nn.BatchNorm1d(patch_size ** 2 * input_channels)
        self.gru_bn_2 = nn.BatchNorm1d((patch_size ** 2) * input_channels)
        self.gru_bn_3 = nn.BatchNorm1d(emb_size *(patch_size ** 2))
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(patch_size ** 2 * input_channels, n_classes)
        self.fc_2 = nn.Linear((patch_size ** 2) * input_channels, n_classes)
        self.fc_3 = nn.Linear(emb_size *(patch_size ** 2) , n_classes)
        self.reg = nn.Linear(emb_size *(patch_size ** 2) , input_channels)
        self.softmax = nn.Softmax()
        self.point_conv_1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1)
        self.depth_conv_1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, groups=input_channels)
        # self.hyper = HypergraphConv(emb_size, emb_size)
        self.num_hper_e = nn.Parameter(torch.randn(1))
        self.aux_loss_weight = 1

    def forward(self, x): #初始是第1方向
        # x1_1 = self.conv_1x1(x)
        # x3_3 = self.conv_3x3(x)
        # x5_5 = self.conv_5_5(x)
        # x = torch.cat([x3_3, x5_5, x1_1], dim=1)
        x = x.squeeze(1)
        # vis.images(x[1, [12,42,55], :, :],opts={'title':'image'})

        # x = self.relu(self.point_conv_1(x)) + self.relu(self.depth_conv_1(x)) + x

        # vis.images(x[1, [12,42,55], :, :],opts={'title':'image'})
        # print('0', x.shape)
        x1 = x
        x1r = x1.reshape(x1.shape[0], x1.shape[1], -1)


        # x2 = Variable(x1r.cpu())
        # x2 = Variable(x1r).cpu()
        x2 = x1r.cpu()
        x2rn = np.flip(x2.detach().numpy(), axis=2).copy()
        x2rt = torch.from_numpy(x2rn)
        x2r = x2rt.cuda()

        x3 = torch.transpose(x1, 2, 3)
        x3r = x3.reshape(x3.shape[0], x3.shape[1], -1)

        # x4 = Variable(x3r.cpu())
        # x4 = Variable(x3r).cpu()
        x4 = x3r.cpu()
        x4rn = np.flip(x4.detach().numpy(), axis=2).copy()
        x4rt = torch.from_numpy(x4rn)
        x4r = x4rt.cuda()

        x5 = torch.rot90(x1, 1, (2, 3))
        x5r = x5.reshape(x5.shape[0], x5.shape[1], -1)

        # x6 = Variable(x5r.cpu())
        # x6 = Variable(x5r).cpu()
        x6 = x5r.cpu()
        x6rn = np.flip(x6.detach().numpy(), axis=2).copy()
        x6rt = torch.from_numpy(x6rn)
        x6r = x6rt.cuda()

        x7 = torch.transpose(x5, 2, 3)
        x7r = x7.reshape(x7.shape[0], x7.shape[1], -1)

        # x8 = Variable(x7r.cpu())
        # x8 = Variable(x7r).cpu()
        x8 = x7r.cpu()
        x8rn = np.flip(x8.detach().numpy(), axis=2).copy()
        x8rt = torch.from_numpy(x8rn)
        x8r = x8rt.cuda()

        x8r = x8r.permute(2, 0, 1)
        x7r = x7r.permute(2, 0, 1)
        x6r = x6r.permute(2, 0, 1)
        x5r = x5r.permute(2, 0, 1)
        x4r = x4r.permute(2, 0, 1)
        x3r = x3r.permute(2, 0, 1)
        x2r = x2r.permute(2, 0, 1)
        x1r = x1r.permute(2, 0, 1)

        x1r = self.pre_emd(x1r)
        x2r = self.pre_emd(x2r)
        x3r = self.pre_emd(x3r)
        x4r = self.pre_emd(x4r)
        x5r = self.pre_emd(x5r)
        x6r = self.pre_emd(x6r)
        x7r = self.pre_emd(x7r)
        x8r = self.pre_emd(x8r)

        # print('x1r shape_for mask', x1r.shape)

        encoder_layer_1 = nn.TransformerEncoderLayer(d_model=self.emb_size, nhead=1, dim_feedforward=self.emb_size,activation='gelu').to(device='cuda')
        transformer_encoder_1 = nn.TransformerEncoder(encoder_layer_1, num_layers=1, norm=None).to(device='cuda')
        encoder_layer_2 = nn.TransformerEncoderLayer(d_model=self.emb_size, nhead=1, dim_feedforward=self.emb_size,activation='gelu').to(device='cuda')
        transformer_encoder_2 = nn.TransformerEncoder(encoder_layer_2, num_layers=1, norm=None).to(device='cuda')
        encoder_layer_3 = nn.TransformerEncoderLayer(d_model=self.emb_size, nhead=1, dim_feedforward=self.emb_size,activation='gelu').to(device='cuda')
        transformer_encoder_3 = nn.TransformerEncoder(encoder_layer_3, num_layers=1, norm=None).to(device='cuda')
        encoder_layer_4 = nn.TransformerEncoderLayer(d_model=self.emb_size, nhead=1, dim_feedforward=self.emb_size,activation='gelu').to(device='cuda')
        transformer_encoder_4 = nn.TransformerEncoder(encoder_layer_4, num_layers=1, norm=None).to(device='cuda')
        encoder_layer_5 = nn.TransformerEncoderLayer(d_model=self.emb_size, nhead=1, dim_feedforward=self.emb_size,activation='gelu').to(device='cuda')
        transformer_encoder_5 = nn.TransformerEncoder(encoder_layer_5, num_layers=1, norm=None).to(device='cuda')
        encoder_layer_6 = nn.TransformerEncoderLayer(d_model=self.emb_size, nhead=1, dim_feedforward=self.emb_size,activation='gelu').to(device='cuda')
        transformer_encoder_6 = nn.TransformerEncoder(encoder_layer_6, num_layers=1, norm=None).to(device='cuda')
        encoder_layer_7 = nn.TransformerEncoderLayer(d_model=self.emb_size, nhead=1, dim_feedforward=self.emb_size,activation='gelu').to(device='cuda')
        transformer_encoder_7 = nn.TransformerEncoder(encoder_layer_7, num_layers=1, norm=None).to(device='cuda')
        encoder_layer_8 = nn.TransformerEncoderLayer(d_model=self.emb_size, nhead=1, dim_feedforward=self.emb_size,activation='gelu').to(device='cuda')
        transformer_encoder_8 = nn.TransformerEncoder(encoder_layer_8, num_layers=1, norm=None).to(device='cuda')

        # 'soft mask with multiscanning'
        # def softweight(x):
        #     x_weight = rearrange(x, 'x b c -> b x c')
        #     x_dist = torch.cdist(x_weight, x_weight, p=2)
        #     mean_x_dist = torch.mean(x_dist)
        #     x_weight_1 = torch.exp(-(x_dist ** 2) / 2 * (mean_x_dist ** 2))
        #     # g = sns.heatmap(x_weight_1[1, :, :].cpu().detach().numpy(), cmap='Blues', square=True)
        #     # g.set_title('')
        #     # plt.show()
        #
        #     mask = np.zeros_like(x_weight_1[1,:,:].cpu().detach().numpy())
        #     mask[np.triu_indices_from(mask)] = True
        #     return x_weight_1
        #
        # mask1 = np.zeros_like(softweight(x1r)[1, :, :].cpu().detach().numpy())
        # mask1[np.triu_indices_from(mask1)] = True
        # plt.subplot(241)
        # g1 = sns.heatmap(softweight(x1r)[1, :, :].cpu().detach().numpy(), cmap='Blues', square=True,mask=mask1)
        # plt.subplot(242)
        # g2 = sns.heatmap(softweight(x2r)[1, :, :].cpu().detach().numpy(), cmap='Blues', square=True)
        # plt.subplot(243)
        # g3 = sns.heatmap(softweight(x3r)[1, :, :].cpu().detach().numpy(), cmap='Blues', square=True)
        # plt.subplot(244)
        # g4 = sns.heatmap(softweight(x4r)[1, :, :].cpu().detach().numpy(), cmap='Blues', square=True)
        # plt.subplot(245)
        # g5 = sns.heatmap(softweight(x5r)[1, :, :].cpu().detach().numpy(), cmap='Blues', square=True)
        # plt.subplot(246)
        # g6 = sns.heatmap(softweight(x6r)[1, :, :].cpu().detach().numpy(), cmap='Blues', square=True)
        # plt.subplot(247)
        # g7 = sns.heatmap(softweight(x7r)[1, :, :].cpu().detach().numpy(), cmap='Blues', square=True)
        # plt.subplot(248)
        # g8 = sns.heatmap(softweight(x8r)[1, :, :].cpu().detach().numpy(), cmap='Blues', square=True)
        # plt.show()


        # '做个spectral soft attention, soft attention mask 速度会变慢'
        # x1r_weight = rearrange(x1r, 'x b c -> b x c')
        # x_dist_1 = torch.cdist(x1r_weight, x1r_weight, p=2)
        # mean_x_dist_1 = torch.mean(x_dist_1)
        # # sns.heatmap(x_dist_1[1, :, :].cpu().detach().numpy(),cmap='Blues',square=True)
        # # # plt.colorbar()
        # # plt.show()
        # x_weight_1 = torch.exp(-(x_dist_1 ** 2) / 2 * (mean_x_dist_1 ** 2))
        # # print('x_weight',x_weight_1.shape)
        # # x_weight = repeat(x_weight,'')
        # # weight = self.sigmoid(weight) * 2
        # g = sns.heatmap(x_weight_1[1, :, :].cpu().detach().numpy(), cmap='Blues',square=True)
        # g.set_title('weight_1')
        # # plt.imshow(weight[1, :, :].cpu().detach().numpy(),cmap='blues')
        # # # plt.colorbar()
        # plt.show()

        # print('x1r',x1r.shape)
        # x = torch.cat([x1r, x1r, x1r, x1r, x1r, x1r, x1r, x1r], dim=2)
        # x1r = self.gru(x1r)[0]

        '---------------hypergraph-----------------'
        # def hyper(x_out):
        #     for b in range(x_out.size(1)):
        #         X = x_out[:,b,:]
        #         hg = dhg.Hypergraph.from_feature_kNN(X, k=5)
        #         X_ = hg.smoothing_with_HGNN(X)
        #         Y_b = hg.v2e(X_,aggr='mean')
        #         X_b = hg.e2v(Y_b,aggr='mean')
        #         if b == 0:
        #             X_new = X_b.unsqueeze(1)
        #         else:
        #             X_new = torch.cat([X_new,X_b.unsqueeze(1)],dim=1)
        #         b = b + 1
        #     print('x_new_2',X_new.shape)
        #     return X_new


        '--------------------------------------------------------------------------------------'
        x1r_out = self.gru_3_1(x1r)
        # print('x1r_output.shape', x1r_out.shape)
        # print('x1r out', x1r_out.shape)  # （25,100,25）
        # print('x1r hidden',x1r_hidden.shape) #（1,100,25）
        # x1r_laststep = x1r_out[-1] #（100,50)
        # print('x1r laststep',x1r_laststep.shape)

        # 'calculate cosine similarity 1'
        # input1 = x1r_out[:,1,:]
        # input_last1 = x1r_laststep[1,:]
        # input_last1 = input_last1.unsqueeze(0)
        # pairdistance = nn.PairwiseDistance(p=2)
        # cossim = nn.CosineSimilarity(dim=1)
        # output1 = pairdistance(input1,input_last1) * (-1)
        # output1 = self.softmax(output1)
        # output1 = output1.unsqueeze(0)
        # # plt.plot(output1[0,:].cpu().detach().numpy(), linewidth =1.5)
        # # plt.show()
        # # sns.heatmap(data=output1.cpu().detach().numpy(), cmap="Blues", linewidths=0.2)
        # # plt.show()
        gamma_1 = self.sigmoid(self.alpha_1)
        delta_1 = 1 - gamma_1
        # x1r_out = transformer_encoder_1(delta_1 * x1r_out + gamma_1 * x1r)
        x1r_out = transformer_encoder_1(x1r_out + x1r) + x1r_out
        # x1r_out = hyper(x1r_out).to(device='cuda')
        # print('x1r_output2.shape', x1r_out.shape) #(step, batch , fea dim)
        # print('gamma_1:', gamma_1, 'delta_1:', delta_1)
        '--------------------------------------------------------------------------------------'
        x2r_out = self.gru_3_2(x2r) #把x1r经过RNN的值，作为x2r的输入
        # x2r_laststep = x2r_out[-1]
        # plt.subplot(2, 4, 2)
        # plt.plot(x2r_laststep[0, :].cpu().detach().numpy())
        # plt.subplot(1, 8, 2)
        # plt.plot(x2r[:, 0, :].cpu().detach().numpy())
        # 'calculate cosine similarity 2'
        # input2 = x2r_out[:, 1, :]
        # input_last2 = x2r_laststep[1, :]
        # input_last2 = input_last2.unsqueeze(0)
        # pairdistance = nn.PairwiseDistance(p=2)
        # cossim = nn.CosineSimilarity(dim=1)
        # output2 = pairdistance(input2, input_last2) * (-1)
        # output2 = self.softmax(output2)
        # output2 = output2.unsqueeze(0)
        gamma_2 = self.sigmoid(self.alpha_2)
        delta_2 = 1 - gamma_2
        # x2r_out = transformer_encoder_2(delta_2 * x2r_out+ gamma_2 * x2r)
        x2r_out = transformer_encoder_2(x2r_out + x2r) +x2r_out
        # x2r_out = hyper(x2r_out).to(device='cuda')
        # print('gamma_2:', gamma_2, 'delta_2:', delta_2)
        '-----------------------------------------------------------------------------------------'
        x3r_out = self.gru_3_3(x3r)
        # x3r_laststep = x3r_out[-1]
        # plt.subplot(2, 4, 3)
        # plt.plot(x3r_laststep[0, :].cpu().detach().numpy())
        # plt.subplot(1, 8, 3)
        # plt.plot(x3r[:, 0, :].cpu().detach().numpy())
        # 'calculate cosine similarity 3'
        # input3 = x3r_out[:, 1, :]
        # input_last3 = x3r_laststep[1, :]
        # input_last3 = input_last3.unsqueeze(0)
        # pairdistance = nn.PairwiseDistance(p=2)
        # cossim = nn.CosineSimilarity(dim=1)
        # output3 = pairdistance(input3, input_last3) * (-1)
        # output3 = self.softmax(output3)
        # output3 = output3.unsqueeze(0)
        gamma_3 = self.sigmoid(self.alpha_3)
        delta_3 = 1 - gamma_3
        # x3r_out = transformer_encoder_3(delta_3 * x3r_out + gamma_3 * x3r)
        x3r_out = transformer_encoder_3(x3r_out + x3r) + x3r_out
        # x3r_out = hyper(x3r_out).to(device='cuda')
        # print('gamma_3:', gamma_3, 'delta_3:', delta_3)
        '----------------------------------------------------------------------------------------'
        x4r_out = self.gru_3_4(x4r)
        # x4r_laststep = x4r_out[-1]
        # plt.subplot(2, 4, 4)
        # plt.plot(x4r_laststep[0, :].cpu().detach().numpy())
        # plt.subplot(1, 8, 4)
        # plt.plot(x4r[:, 0, :].cpu().detach().numpy())
        # 'calculate cosine similarity 4'
        # input4 = x4r_out[:, 1, :]
        # input_last4 = x4r_laststep[1, :]
        # input_last4 = input_last4.unsqueeze(0)
        # pairdistance = nn.PairwiseDistance(p=2)
        # cossim = nn.CosineSimilarity(dim=1)
        # output4 = pairdistance(input4, input_last4) * (-1)
        # output4 = self.softmax(output4)
        # output4 = output4.unsqueeze(0)
        gamma_4 = self.sigmoid(self.alpha_4)
        delta_4 = 1 - gamma_4
        # x4r_out = transformer_encoder_4(delta_4 * x4r_out + gamma_4 * x4r)
        x4r_out = transformer_encoder_4(x4r_out + x4r) +x4r_out
        # x4r_out = hyper(x4r_out).to(device='cuda')
        # print('gamma_4:', gamma_4, 'delta_4:', delta_4)
        '------------------------------------------------------------------------------------------'
        x5r_out = self.gru_3_5(x5r)
        # x5r_laststep = x5r_out[-1]
        # plt.subplot(2, 4, 5)
        # plt.plot(x5r_laststep[0, :].cpu().detach().numpy())
        # plt.subplot(1, 8, 5)
        # plt.plot(x5r[:, 0, :].cpu().detach().numpy())
        # 'calculate cosine similarity 5'
        # input5 = x5r_out[:, 1, :]
        # input_last5 = x5r_laststep[1, :]
        # input_last5 = input_last5.unsqueeze(0)
        # pairdistance = nn.PairwiseDistance(p=2)
        # cossim = nn.CosineSimilarity(dim=1)
        # output5 = pairdistance(input5, input_last5) * (-1)
        # output5 = self.softmax(output5)
        # output5 = output5.unsqueeze(0)
        gamma_5 = self.sigmoid(self.alpha_5)
        delta_5 = 1 - gamma_5
        # x5r_out = transformer_encoder_5(delta_5 * x5r_out + gamma_5 * x5r)
        x5r_out = transformer_encoder_5(x5r_out + x5r) +x5r_out
        # x5r_out = hyper(x5r_out).to(device='cuda')
        # print('gamma_5:', gamma_5, 'delta_5:', delta_5)
        '------------------------------------------------------------------------------------------'
        x6r_out= self.gru_3_6(x6r)
        # x6r_laststep = x6r_out[-1]
        # plt.subplot(2, 4, 6)
        # plt.plot(x6r_laststep[0, :].cpu().detach().numpy())
        # plt.subplot(1, 8, 6)
        # plt.plot(x6r[:, 0, :].cpu().detach().numpy())
        # 'calculate cosine similarity 6'
        # input6 = x6r_out[:, 1, :]
        # input_last6 = x6r_laststep[1, :]
        # input_last6 = input_last6.unsqueeze(0)
        # pairdistance = nn.PairwiseDistance(p=2)
        # cossim = nn.CosineSimilarity(dim=1)
        # output6 = pairdistance(input6, input_last6) * (-1)
        # output6 = self.softmax(output6)
        # output6 = output6.unsqueeze(0)
        gamma_6 = self.sigmoid(self.alpha_6)
        delta_6 = 1 - gamma_6
        # x6r_out = transformer_encoder_6(delta_6 * x6r_out + gamma_6 * x6r)
        x6r_out = transformer_encoder_6(x6r_out + x6r) +x6r_out
        # x6r_out = hyper(x6r_out).to(device='cuda')
        # print('gamma_6:', gamma_6, 'delta_6:', delta_6)
        '---------------------------------------------------------------------------------------------'
        x7r_out= self.gru_3_7(x7r)
        # x7r_laststep = x7r_out[-1]
        # plt.subplot(2, 4, 7)
        # plt.plot(x7r_laststep[0, :].cpu().detach().numpy())
        # plt.subplot(1, 8, 7)
        # plt.plot(x7r[:, 0, :].cpu().detach().numpy())
        # 'calculate cosine similarity 7'
        # input7 = x7r_out[:, 1, :]
        # input_last7 = x7r_laststep[1, :]
        # input_last7 = input_last7.unsqueeze(0)
        # pairdistance = nn.PairwiseDistance(p=2)
        # cossim = nn.CosineSimilarity(dim=1)
        # output7 = pairdistance(input7, input_last7) * (-1)
        # output7 = self.softmax(output7)
        # output7 = output7.unsqueeze(0)
        gamma_7 = self.sigmoid(self.alpha_7)
        delta_7 = 1 - gamma_7
        # x7r_out = transformer_encoder_7(delta_7 * x7r_out + gamma_7 * x7r)
        x7r_out = transformer_encoder_7(x7r_out + x7r) +x7r_out
        # x7r_out = hyper(x7r_out).to(device='cuda')
        # print('gamma_7:', gamma_7, 'delta_7:', delta_7)
        '----------------------------------------------------------------------------------------------'
        x8r_out= self.gru_3_8(x8r)
        # x8r_laststep = x8r_out[-1]
        # ax8 = plt.subplot(2, 4, 8)
        # ax8.set_title('8')
        # plt.plot(x8r_laststep[0,:].cpu().detach().numpy())
        # plt.subplot(1, 8, 8)
        # plt.plot(x8r[:, 0, :].cpu().detach().numpy())
        # # x8r = self.gru(x8r+x7r)[0]
        # print('x8r_out',x8r_out.shape)
        # 'calculate cosine similarity 8'
        # input8 = x8r_out[:, 1, :]
        # input_last8 = x8r_laststep[1, :]
        # input_last8 = input_last8.unsqueeze(0)
        # pairdistance = nn.PairwiseDistance(p=2)
        # cossim = nn.CosineSimilarity(dim=1)
        # output8 = pairdistance(input8, input_last8) * (-1)
        # output8 = self.softmax(output8)
        # output8 = output8.unsqueeze(0)
        gamma_8 = self.sigmoid(self.alpha_8)
        delta_8 = 1 - gamma_8
        # x8r_out = transformer_encoder_8(delta_8 * x8r_out + gamma_8 * x8r)
        x8r_out = transformer_encoder_8(x8r_out + x8r) +x8r_out  # (b n d)
        # x8r_out = hyper(x8r_out).to(device='cuda')
        # print('gamma_8:', gamma_8, 'delta_8:', delta_8)
        step = int(x8r_out.size(0)-1)
        '-------------------------------------------------------------------------------'
        '----show attetntion function------------------------------------------------------'
        # def showattention(inputseq):
        #     allpixel = inputseq[:, 1, :]
        #     linear1 = nn.Linear(allpixel.size(1), allpixel.size(1)).to(device='cuda')
        #     allpixel = linear1(allpixel)
        #
        #     # centralstep = allpixel[12,:]
        #     # laststep = inputseq[int(step/2), 1, :]
        #     # laststep = linear1(laststep)
        #
        #     # centralstep = allpixel[12,:]
        #     centralstep = allpixel[int(step / 2), :]
        #     # centralstep = linear1(centralstep)
        #
        #     pairdis = nn.PairwiseDistance()
        #     cos = nn.CosineSimilarity(dim=-1)
        #
        #     output = torch.matmul(allpixel, centralstep)
        #     # output = pairdis(allpixel, centralstep) * (-1)
        #     # output = cos(allpixel, centralstep) * (-1)
        #
        #     # output = torch.matmul(allpixel, centralstep)
        #     softmax = nn.Softmax()
        #     output = softmax(output)
        #     output = output.unsqueeze(0)
        #     return output
        #
        # '------------------------------------------------------------------------------------'
        # print('x1r_out.shape',x1r_out.shape)
        # output1_1 = showattention(x1r_out)
        # print('......',output1_1.shape)
        # output1_1_image = reduce(output1_1, 'v (h w) -> h w', h=self.patch_size,reduction='mean')
        # sns.heatmap(data=output1_1_image[:,:].cpu().detach().numpy())
        # plt.show()
        # print('......', output1_1.shape)
        # output2_2 = showattention(x2r_out)
        # output3_3 = showattention(x3r_out)
        # output4_4 = showattention(x4r_out)
        # output5_5 = showattention(x5r_out)
        # output6_6 = showattention(x6r_out)
        # output7_7 = showattention(x7r_out)
        # output8_8 = showattention(x8r_out)
        # # '----------------------------------------------------------------------------'
        # outputall = torch.cat([output1_1, output2_2, output3_3, output4_4, output5_5, output6_6, output7_7, output8_8],dim=0)
        # sns.lineplot(data=outputall.transpose(1,0).cpu().detach().numpy(), markers=True, lw=2)
        # plt.show()
        # all = sns.heatmap(data=outputall.cpu().detach().numpy(), cmap="Blues", linewidths=0.05)
        # all.set_title('all')
        # plt.show()

        '--------------------------------------------------------------------------------------------------'

        # b = x8r_out.shape[1]

        # decoder_layer_1 = nn.TransformerDecoderLayer(d_model=50, nhead=1, dim_feedforward=64, activation='gelu').to(
        #     device='cuda')
        # transformer_decoder_1 = nn.TransformerDecoder(decoder_layer_1, num_layers=1, norm=None).to(device='cuda')
        # memory_1 = x1r_out
        # target_1 = repeat(x1r_out[12,:,:], 'b c -> 1 b c')
        # x1r_decoder_out = transformer_decoder_1(target_1,memory_1)
        # # plt.subplot(121)
        # # plt.imshow(x1r_output[:,1,:].cpu().detach().numpy())
        # # plt.subplot(122)
        # # plt.imshow(x1r_decoder_out[:,1,:].cpu().detach().numpy())
        # # plt.show()
        # decoder_layer_2 = nn.TransformerDecoderLayer(d_model=50, nhead=1, dim_feedforward=64, activation='gelu').to(
        #     device='cuda')
        # transformer_decoder_2 = nn.TransformerDecoder(decoder_layer_2, num_layers=1, norm=None).to(device='cuda')
        # memory_2 = x2r_out
        # target_2 = repeat(x2r_out[12,:,:], 'b c -> 1 b c')
        # x2r_decoder_out = transformer_decoder_2(target_2, memory_2)
        #
        # decoder_layer_3 = nn.TransformerDecoderLayer(d_model=50, nhead=1, dim_feedforward=64, activation='gelu').to(
        #     device='cuda')
        # transformer_decoder_3 = nn.TransformerDecoder(decoder_layer_3, num_layers=1, norm=None).to(device='cuda')
        # memory_3 = x3r_out
        # target_3 = repeat(x3r_out[12,:,:], 'b c -> 1 b c')
        # x3r_decoder_out = transformer_decoder_3(target_3, memory_3)
        #
        # decoder_layer_4 = nn.TransformerDecoderLayer(d_model=50, nhead=1, dim_feedforward=64, activation='gelu').to(
        #     device='cuda')
        # transformer_decoder_4 = nn.TransformerDecoder(decoder_layer_4, num_layers=1, norm=None).to(device='cuda')
        # memory_4 = x4r_out
        # target_4 = repeat(x4r_out[12,:,:],'b c -> 1 b c')
        # x4r_decoder_out = transformer_decoder_4(target_4, memory_4)
        #
        # decoder_layer_5 = nn.TransformerDecoderLayer(d_model=50, nhead=1, dim_feedforward=64, activation='gelu').to(
        #     device='cuda')
        # transformer_decoder_5 = nn.TransformerDecoder(decoder_layer_5, num_layers=1, norm=None).to(device='cuda')
        # memory_5 = x5r_out
        # target_5 = repeat(x5r_out[12,:,:], 'b c -> 1 b c')
        # x5r_decoder_out = transformer_decoder_5(target_5, memory_5)
        #
        # decoder_layer_6 = nn.TransformerDecoderLayer(d_model=50, nhead=1, dim_feedforward=64, activation='gelu').to(
        #     device='cuda')
        # transformer_decoder_6 = nn.TransformerDecoder(decoder_layer_6, num_layers=1, norm=None).to(device='cuda')
        # memory_6 = x6r_out
        # target_6 = repeat(x6r_out[12,:,:], 'b c -> 1 b c')
        # x6r_decoder_out = transformer_decoder_6(target_6, memory_6)
        #
        # decoder_layer_7 = nn.TransformerDecoderLayer(d_model=50, nhead=1, dim_feedforward=64, activation='gelu').to(
        #     device='cuda')
        # transformer_decoder_7 = nn.TransformerDecoder(decoder_layer_7, num_layers=1, norm=None).to(device='cuda')
        # memory_7 = x7r_out
        # target_7 = repeat(x7r_out[12,:,:], 'b c -> 1 b c')
        # x7r_decoder_out = transformer_decoder_7(target_7, memory_7)
        #
        # decoder_layer_8 = nn.TransformerDecoderLayer(d_model=50, nhead=1, dim_feedforward=64, activation='gelu').to(
        #     device='cuda')
        # transformer_decoder_8 = nn.TransformerDecoder(decoder_layer_8, num_layers=1, norm=None).to(device='cuda')
        # memory_8 = x8r_out
        # target_8 = repeat(x8r_out[12,:,:], 'b c -> 1 b c')
        # x8r_decoder_out = transformer_decoder_8(target_8, memory_8)

        b = x1r_out.size(1)
        # print('b',b)
        alpha_1 = (repeat(self.alpha_1, '1 1 -> b 1',b=b))
        alpha_2 = (repeat(self.alpha_2, '1 1 -> b 1',b=b))
        alpha_3 = (repeat(self.alpha_3, '1 1 -> b 1',b=b))
        alpha_4 = (repeat(self.alpha_4, '1 1 -> b 1',b=b))
        alpha_5 = (repeat(self.alpha_5, '1 1 -> b 1',b=b))
        alpha_6 = (repeat(self.alpha_6, '1 1 -> b 1',b=b))
        alpha_7 = (repeat(self.alpha_7, '1 1 -> b 1',b=b))
        alpha_8 = (repeat(self.alpha_8, '1 1 -> b 1',b=b))
        # print('alpha',alpha_1.shape)
        # alpha = alpha_1 + alpha_2 + alpha_3 + alpha_4 + alpha_5 + alpha_6 + alpha_7 + alpha_8
        # alpha_1 = alpha_1 / alpha
        # alpha_2 = alpha_2 / alpha
        # alpha_3 = alpha_3 / alpha
        # alpha_4 = alpha_4 / alpha
        # alpha_5 = alpha_5 / alpha
        # alpha_6 = alpha_6 / alpha
        # alpha_7 = alpha_7 / alpha
        # alpha_8 = alpha_8 / alpha
        attn_alphs = self.softmax(torch.cat([alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6, alpha_7, alpha_8], dim=1))
        # print('attn_alpha',attn_alphs.shape)
        # plt.plot(attn_alphs[1,:].cpu().detach().numpy())
        # plt.show()
        attn_alphs = reduce(attn_alphs,'b s -> s', reduction='mean')
        # alpha_1 = rearrange(attn_alphs[:,0],'')

        # if alpha_1 + alpha_2 + alpha_3 + alpha_4 + alpha_5 + alpha_6 + alpha_7 + alpha_8 == 1:
        x = x8r_out + x7r_out + x6r_out + x5r_out + x4r_out+ x3r_out + x2r_out + x1r_out
        # x = x8r_decoder_out + x7r_decoder_out + x6r_decoder_out + x5r_decoder_out + x4r_decoder_out + x3r_decoder_out + x2r_decoder_out + x1r_decoder_out
        # x = x8r_out * attn_alphs[7]  + x7r_out*attn_alphs[6] + x6r_out*attn_alphs[5] + x5r_out*attn_alphs[4] + x4r_out*attn_alphs[3] + x3r_out*attn_alphs[2] + x2r_out * attn_alphs[1] + x1r_out * attn_alphs[0]

        # print('a1:',alpha_1,'a2:',alpha_2,'a3:',alpha_3,'a4:',alpha_4,'a5:',alpha_5,'a6:',alpha_6,'a7:',alpha_7,'a8:',alpha_8)



        # x = torch.cat([x1r,x2r,x3r,x4r,x5r,x6r,x7r,x8r],dim=2)
        # x = self.gru_bn(x)
        # x = x1r + x2r + x3r + x4r + x5r + x6r + x7r + x8r
        # print('x',x.shape)
        # print('into GRU',x3.shape)
        # x4 = self.gru(x4)[0]
        # x3 = self.gru(x3)[0]
        # x2 = self.gru(x2)[0]

        # x = self.gru(x)[0]
        # x = self.gru2(x)[0]

        # print('out GRU',x3.shape)
        # x4 = x4.permute(1, 2, 0).contiguous()
        # x3 = x3.permute(1, 2, 0).contiguous()
        # x2 = x2.permute(1, 2, 0).contiguous()
        # x1 = x1.permute(1, 2, 0).contiguous()
        x = x.permute(1,2,0).contiguous()
        # print('5-1',x1.shape)

        # x4 = x4.view(x4.size(0), -1)
        # x3 = x3.view(x3.size(0), -1)
        # x2 = x2.view(x2.size(0), -1)
        # x1 = x1.view(x1.size(0), -1)
        x = x.view(x.size(0),-1)
        # print('x',x.shape)

        # x = x4 + x3 + x2 + x1
        # # w1 = x1 / x
        # # w2 = x2 / x
        # # w3 = x3 / x
        # # w4 = x4 / x
        # x = 0.35*x1 + 0.35*x2 + 0.15*x3 +0.15*x4
        # # x = w1*x1 + w2*x2 + w3*x3 + w4*x4
        # print('into gru_bn', x.shape)
        x = self.gru_bn_3(x)
        # x = self.gru_bn2(x)
        # x = self.relu(x)
        x = self.tanh(x)
        # x = self.elu(x)
        # x =self.prelu(x)
        # print('into fc',x.shape)
        x = self.dropout(x)
        x_class = self.fc_3(x)
        x_reg = self.reg(x)
        # plt.show()
        # x = self.fc2(x)
        return x_class, x_reg

class zhouEightDRNN_kamata_LSTM(nn.Module):

    @staticmethod
    def weight_init(m):
        # All weight matrices in our RNN and bias vectors are initialized with a uniform distribution, and the values of these weight matrices and bias vectors are initialized in the range [−0.1,0.1]
        if isinstance(m, (nn.Linear, nn.GRU, nn.Conv3d,nn.LSTM)):
            init.kaiming_uniform_(m.weight.data)
            init.kaiming_uniform_(m.bias.data)

    def __init__(self, input_channels, n_classes, patch_size=5):
        # The proposed network model uses a single recurrent layer that adopts our modified GRUs of size 64 with sigmoid gate activation and PRetanh activation functions for hidden representations
        super(zhouEightDRNN_kamata_LSTM, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.gru = nn.GRU(patch_size ** 2, patch_size ** 2, 1, bidirectional=False,
                          batch_first=False)  # TODO: try to change this ?
        self.gru_2 = nn.GRU(input_channels, input_channels, 1, bidirectional=False)
        self.gru_2_1 = nn.GRU(input_channels, patch_size**2, 1, bidirectional=False)
        self.gru_2_2 = nn.GRU(input_channels, patch_size**2, 1, bidirectional=False)
        self.gru_2_3 = nn.GRU(input_channels, patch_size**2, 1, bidirectional=False)
        self.gru_2_4 = nn.GRU(input_channels, patch_size**2, 1, bidirectional=False)
        self.gru_2_5 = nn.GRU(input_channels, patch_size**2, 1, bidirectional=False)
        self.gru_2_6 = nn.GRU(input_channels, patch_size**2, 1, bidirectional=False)
        self.gru_2_7 = nn.GRU(input_channels, patch_size**2, 1, bidirectional=False)
        self.gru_2_8 = nn.GRU(input_channels, patch_size**2, 1, bidirectional=False)
        self.lstm_2_1 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.lstm_2_2 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.lstm_2_3 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.lstm_2_4 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.lstm_2_5 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.lstm_2_6 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.lstm_2_7 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.lstm_2_8 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        # self.lstm_2_1 = nn.RNN(input_channels, 64, 1, bidirectional=False)
        # self.lstm_2_2 = nn.RNN(input_channels, 64, 1, bidirectional=False)
        # self.lstm_2_3 = nn.RNN(input_channels, 64, 1, bidirectional=False)
        # self.lstm_2_4 = nn.RNN(input_channels, 64, 1, bidirectional=False)
        # self.lstm_2_5 = nn.RNN(input_channels, 64, 1, bidirectional=False)
        # self.lstm_2_6 = nn.RNN(input_channels, 64, 1, bidirectional=False)
        # self.lstm_2_7 = nn.RNN(input_channels, 64, 1, bidirectional=False)
        # self.lstm_2_8 = nn.RNN(input_channels, 64, 1, bidirectional=False)
        self.lstm_stra_1 = nn.LSTM(64, 64, 1, bidirectional=False)
        # self.gru_3 = nn.GRU(input_channels, patch_size ** 2, 1, bidirectional=True)
        # self.gru_3_1 = nn.GRU(input_channels, patch_size ** 2, 1, bidirectional=True)
        # self.gru_3_2 = nn.GRU(input_channels, patch_size ** 2, 1, bidirectional=True)
        # self.gru_3_3 = nn.GRU(input_channels, patch_size ** 2, 1, bidirectional=True)
        # self.gru_3_4 = nn.GRU(input_channels, patch_size ** 2, 1, bidirectional=True)
        # self.gru_3_5 = nn.GRU(input_channels, patch_size ** 2, 1, bidirectional=True)
        # self.gru_3_6 = nn.GRU(input_channels, patch_size ** 2, 1, bidirectional=True)
        # self.gru_3_7 = nn.GRU(input_channels, patch_size ** 2 , 1, bidirectional=True)
        # self.gru_3_8 = nn.GRU(input_channels, patch_size ** 2, 1, bidirectional=True)
        self.scan_order = Order_Attention(64, 64)

        self.gru_4 = nn.GRU(64, 64, 1)
        self.lstm_4 = nn.LSTM(patch_size ** 2, 64, 1)
        self.conv = nn.Conv2d(input_channels,out_channels=input_channels, kernel_size=(3,3),stride=(3,3))
        self.transformer = nn.Transformer(nhead=16, num_encoder_layers=12)
        self.gru_bn = nn.BatchNorm1d(patch_size ** 2 * input_channels)
        self.lstm_bn_1 = nn.BatchNorm1d((64)*1)
        self.lstm_bn_1_2 = nn.BatchNorm1d((64) * (patch_size)**2)
        self.lstm_bn_2 = nn.BatchNorm1d((64)*8)
        self.lstm_bn_2_2 = nn.BatchNorm1d((64) * 8 * patch_size**2)
        self.gru_bn_2 = nn.BatchNorm1d((patch_size ** 2) * (patch_size**2))
        self.gru_bn_3 = nn.BatchNorm1d((patch_size ** 2) * (patch_size ** 2) * 2)
        self.gru_bn_4 = nn.BatchNorm1d(8 * 64)
        self.gru_bn_laststep = nn.BatchNorm1d(64)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(patch_size ** 2 * input_channels, n_classes)
        self.lstm_fc_1 = nn.Linear(64 * 1, n_classes)
        self.lstm_fc_1_2 = nn.Linear(64 * (patch_size**2), n_classes)
        self.lstm_fc_2 = nn.Linear(64*8,n_classes)
        self.lstm_fc_2_2 = nn.Linear(64 * 8 * patch_size**2, n_classes)
        self.fc_2 = nn.Linear((patch_size ** 2) * (patch_size**2), n_classes)
        self.fc_3 = nn.Linear((patch_size ** 2) * (patch_size ** 2) * 2, n_classes)
        self.fc_4 = nn.Linear(8 * 64, n_classes)
        self.fc_laststep = nn.Linear(64, n_classes)
        self.softmax = nn.Softmax()
        self.linear = nn.Linear(64,64)
        self.aux_loss_weight = 1

    def forward(self, x): #初始是第1方向
        print('x.shape1',x.shape)
        x = x.squeeze(1)
        print('x.shape2', x.shape)
        # x = self.conv(x)
        print('x.shape3', x.shape)
        # x_matrix = x[0,:,:,:]
        # x_matrix = x_matrix.cpu()
        # # plt.subplot(331)
        # plt.imshow(x_matrix[0,:,:], interpolation='nearest', origin='upper')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # plt.xticks(fontsize=25)
        # plt.yticks(fontsize=25)
        # plt.xlabel('X-Position', fontdict={'size': 25}, fontweight='bold')
        # plt.ylabel('Y-Position', fontdict={'size': 25}, fontweight='bold')
        # plt.title('Values of last dimension in the patch',fontdict={'size': 20}, fontweight='bold')
        # plt.show()

        #生成第1和7
        x1_0 = x[:, :, 0, :]
        x1_1 = x[:, :, 1, :]
        x1_2 = x[:, :, 2, :]
        x1_3 = x[:, :, 3, :]
        x1_4 = x[:, :, 4, :]
        x1_1f = torch.flip(x1_1, [2])
        x1_3f = torch.flip(x1_3, [2])
        # plt.subplot(3, 4, 9).set_title('Spectral signatures in a patch')
        # direction_1_showpicture = torch.cat([x1_0, x1_1f, x1_2, x1_3f, x1_4], dim=2)
        # plt.xlabel('Band Numbers', fontdict={'size': 15}, fontweight='bold')
        # plt.ylabel('Spectral Values Values', fontdict={'size': 15},fontweight='bold')
        # plt.plot(direction_1_showpicture[0, :, :].cpu().detach().numpy())
        # direction_1 = torch.cat([x1_0, x1_1f, x1_2], dim=2)
        direction_1 = torch.cat([x1_0, x1_1f, x1_2,x1_3f,x1_4], dim=2)

        print('d1',direction_1.shape)
        # print('d1',direction_1.shape)
        direction_7 = torch.flip(direction_1,[2])

        #生成第2和8
        x2_0 = x[:, :, :, 0]
        x2_1 = x[:, :, :, 1]
        x2_2 = x[:, :, :, 2]
        x2_3 = x[:, :, :, 3]
        x2_4 = x[:, :, :, 4]
        x2_1f = torch.flip(x2_1, [2])
        x2_3f = torch.flip(x2_3, [2])
        # direction_2 = torch.cat([x2_0, x2_1f, x2_2], dim=2)
        direction_2 = torch.cat([x2_0, x2_1f,x2_2,x2_3f,x2_4], dim=2)
        direction_8 = torch.flip(direction_2, [2])

        # 生成3和5
        x3_0 = x[:, :, 0, :]
        x3_1 = x[:, :, 1, :]
        x3_2 = x[:, :, 2, :]
        x3_3 = x[:, :, 3, :]
        x3_4 = x[:, :, 4, :]
        x3_0f = torch.flip(x3_0, [2])
        x3_2f = torch.flip(x3_2, [2])
        x3_4f = torch.flip(x3_4, [2])
        # direction_3 = torch.cat([x3_0f, x3_1, x3_2f], dim=2)
        direction_3 = torch.cat([x3_0f, x3_1, x3_2f,x3_3,x3_4f], dim=2)
        direction_5 = torch.flip(direction_3, [2])

        #生成4和6
        x4_0 = x[:, :, :, 0]
        x4_1 = x[:, :, :, 1]
        x4_2 = x[:, :, :, 2]
        x4_3 = x[:, :, :, 3]
        x4_4 = x[:, :, :, 4]
        x4_1f = torch.flip(x4_1, [2])
        x4_3f = torch.flip(x4_3, [2])
        # direction_4 = torch.cat([x4_2, x4_1f, x4_0], dim=2)
        direction_4 = torch.cat([x4_4, x4_3f, x4_2, x4_1f, x4_0], dim=2)
        # print('d4', direction_4.shape)
        direction_6 = torch.flip(direction_4, [2])

        # plt.xlabel('Band Numbers', fontdict={'size': 35}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 35}, fontweight='bold')
        # plt.plot(direction_1[0, :, 0].cpu().detach().numpy(), label='index 0')
        # plt.plot(direction_1[0, :, 1].cpu().detach().numpy(), label='index 1')
        # plt.plot(direction_1[0, :, 2].cpu().detach().numpy(), label='index 2')
        # plt.plot(direction_1[0, :, 3].cpu().detach().numpy(), label='index 3')
        # plt.plot(direction_1[0, :, 4].cpu().detach().numpy(), label='index 4')
        # plt.plot(direction_1[0, :, 5].cpu().detach().numpy(), label='index 9')
        # plt.plot(direction_1[0, :, 6].cpu().detach().numpy(), label='index 8')
        # plt.plot(direction_1[0, :, 7].cpu().detach().numpy(), label='index 7')
        # plt.plot(direction_1[0, :, 8].cpu().detach().numpy(), label='index 6')
        # plt.plot(direction_1[0, :, 9].cpu().detach().numpy(), label='index 5')
        # plt.plot(direction_1[0, :, 10].cpu().detach().numpy(), label='index 10')
        # plt.plot(direction_1[0, :, 11].cpu().detach().numpy(), label='index 11')
        # plt.plot(direction_1[0, :, 12].cpu().detach().numpy(), label='index 12', linewidth=5, linestyle='-.', color = 'red' )
        # plt.plot(direction_1[0, :, 13].cpu().detach().numpy(), label='index 13')
        # plt.plot(direction_1[0, :, 14].cpu().detach().numpy(), label='index 14')
        # plt.plot(direction_1[0, :, 15].cpu().detach().numpy(), label='index 19')
        # plt.plot(direction_1[0, :, 16].cpu().detach().numpy(), label='index 18')
        # plt.plot(direction_1[0, :, 17].cpu().detach().numpy(), label='index 17')
        # plt.plot(direction_1[0, :, 18].cpu().detach().numpy(), label='index 16')
        # plt.plot(direction_1[0, :, 19].cpu().detach().numpy(), label='index 15')
        # plt.plot(direction_1[0, :, 20].cpu().detach().numpy(), label='index 20')
        # plt.plot(direction_1[0, :, 21].cpu().detach().numpy(), label='index 21')
        # plt.plot(direction_1[0, :, 22].cpu().detach().numpy(), label='index 22')
        # plt.plot(direction_1[0, :, 23].cpu().detach().numpy(), label='index 23')
        # plt.plot(direction_1[0, :, 24].cpu().detach().numpy(), label='index 24')
        # plt.legend(prop={'size': 15}, fontsize='large', loc=[1, 0])
        # plt.title('Spectral signatures in a patch', fontdict={'size': 35}, fontweight='bold')
        # plt.grid(linewidth = 1.5)
        # plt.xticks(fontsize=25)
        # plt.yticks(fontsize=25)
        # plt.show()
        # plt.subplot(122)
        # plt.xlabel('Band Numbers', fontdict={'size': 35}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 35}, fontweight='bold')
        # plt.plot(direction_2[0, :, 0].cpu().detach().numpy(), label='(0,0),0')
        # plt.plot(direction_2[0, :, 1].cpu().detach().numpy(), label='(1,0),5')
        # plt.plot(direction_2[0, :, 2].cpu().detach().numpy(), label='(2,0),10')
        # plt.plot(direction_2[0, :, 3].cpu().detach().numpy(), label='(3,0),15')
        # plt.plot(direction_2[0, :, 4].cpu().detach().numpy(), label='(4,0),20')
        # plt.plot(direction_2[0, :, 5].cpu().detach().numpy(), label='(4,1),21')
        # plt.plot(direction_2[0, :, 6].cpu().detach().numpy(), label='(3,1),16')
        # plt.plot(direction_2[0, :, 7].cpu().detach().numpy(), label='(2,1),11')
        # plt.plot(direction_2[0, :, 8].cpu().detach().numpy(), label='(1,1),6')
        # plt.plot(direction_2[0, :, 9].cpu().detach().numpy(), label='(0,1),1')
        # plt.plot(direction_2[0, :, 10].cpu().detach().numpy(), label='(0,2),2')
        # plt.plot(direction_2[0, :, 11].cpu().detach().numpy(), label='(1,2),7')
        # plt.plot(direction_2[0, :, 12].cpu().detach().numpy(), label='(2,2), center', linewidth=3, linestyle='-.')
        # plt.plot(direction_2[0, :, 13].cpu().detach().numpy(), label='(3,2),17')
        # plt.plot(direction_2[0, :, 14].cpu().detach().numpy(), label='(4,2),22')
        # plt.plot(direction_2[0, :, 15].cpu().detach().numpy(), label='(4,3),23')
        # plt.plot(direction_2[0, :, 16].cpu().detach().numpy(), label='(3,3),18')
        # plt.plot(direction_2[0, :, 17].cpu().detach().numpy(), label='(2,3),13')
        # plt.plot(direction_2[0, :, 18].cpu().detach().numpy(), label='(1,3),8')
        # plt.plot(direction_2[0, :, 19].cpu().detach().numpy(), label='(0,3),3', linewidth=5)
        # plt.plot(direction_2[0, :, 20].cpu().detach().numpy(), label='(0,4),4', linewidth=5)
        # plt.plot(direction_2[0, :, 21].cpu().detach().numpy(), label='(1,4),9', linewidth=5)
        # plt.plot(direction_2[0, :, 22].cpu().detach().numpy(), label='(2,4),14')
        # plt.plot(direction_2[0, :, 23].cpu().detach().numpy(), label='(3,4),19')
        # plt.plot(direction_2[0, :, 24].cpu().detach().numpy(), label='(4,4),24')
        # plt.legend(prop={'size': 15}, fontsize='large', loc=[1, 0])
        # plt.title('Spectral signatures in a patch', fontdict={'size': 35}, fontweight='bold')
        # plt.xticks(fontsize=25)
        # plt.yticks(fontsize=25)
        # plt.show()

        # # plt.subplot(332)
        # plt.imshow(direction_1[0, :, (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-1 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(333)
        # plt.imshow(direction_2[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-2 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(334)
        # plt.imshow(direction_3[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-3 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(335)
        # plt.imshow(direction_4[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-4 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(336)
        # plt.imshow(direction_5[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-5 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(337)
        # plt.imshow(direction_6[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-6 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(338)
        # plt.imshow(direction_7[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()ticks(fontsize=20)
        # plt.title('Direction-7 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(339)
        # plt.imshow(direction_8[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-8 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()

        #换成输入顺序
        x8r = direction_8.permute(2, 0, 1)
        x7r = direction_7.permute(2, 0, 1)
        x6r = direction_6.permute(2, 0, 1)
        x5r = direction_5.permute(2, 0, 1)
        x4r = direction_4.permute(2, 0, 1)
        x3r = direction_3.permute(2, 0, 1)
        x2r = direction_2.permute(2, 0, 1)
        x1r = direction_1.permute(2, 0, 1)
        # print('d5.shape', x5r.shape)
        # plt.subplot(3, 4, 9)
        # plt.plot(direction_1[0, :, :].cpu().detach().numpy())

        'soft mask with multiscanning'
        # def softweight(x):
        #     x_weight = rearrange(x, 'x b c -> b x c')
        #     x_dist = torch.cdist(x_weight, x_weight, p=2)
        #     mean_x_dist = torch.mean(x_dist)
        #     x_weight_1 = torch.exp(-(x_dist ** 2) / 2 * (mean_x_dist ** 2))
        #     # g = sns.heatmap(x_weight_1[1, :, :].cpu().detach().numpy(), cmap='Blues', square=True)
        #     # g.set_title('')
        #     # plt.show()
        #     return x_weight_1
        # plt.subplot(241)
        # g1 = sns.heatmap(softweight(x1r)[1, :, :].cpu().detach().numpy(), cmap='Blues', square=True)
        # # cbar = g1.collections[0].colorbar
        # # cbar.ax.tick_params(labelsize=20)
        # plt.subplot(242)
        # g2 = sns.heatmap(softweight(x2r)[1, :, :].cpu().detach().numpy(), cmap='Blues', square=True)
        # plt.subplot(243)
        # g3 = sns.heatmap(softweight(x3r)[1, :, :].cpu().detach().numpy(), cmap='Blues', square=True, )
        # plt.subplot(244)
        # g4 = sns.heatmap(softweight(x4r)[1, :, :].cpu().detach().numpy(), cmap='Blues', square=True,)
        # plt.subplot(245)
        # g5 = sns.heatmap(softweight(x5r)[1, :, :].cpu().detach().numpy(), cmap='Blues', square=True, )
        # plt.subplot(246)
        # g6 = sns.heatmap(softweight(x6r)[1, :, :].cpu().detach().numpy(), cmap='Blues', square=True,)
        # plt.subplot(247)
        # g7 = sns.heatmap(softweight(x7r)[1, :, :].cpu().detach().numpy(), cmap='Blues', square=True,)
        # plt.subplot(248)
        # g8 = sns.heatmap(softweight(x8r)[1, :, :].cpu().detach().numpy(), cmap='Blues', square=True,)
        # plt.show()

        print('x1r',x1r.shape)
        '-----------------------------------------------------------------------------------------------------'
        h0_x1r = torch.zeros(1, x1r.size(1), 64).to(device='cuda')
        c0_x1r = torch.zeros(1, x1r.size(1), 64).to(device="cuda")
        x1r, x1r_hidden = self.lstm_2_1(x1r)

        # print('hidden', x1r_hidden.shape)
        # x1r = self.gru_2_1(x1r)[0]
        print('x1r', x1r.shape)
        x1r_laststep = x1r[-1]
        x1r_laststep = self.relu(x1r_laststep)
        x1r_laststep = torch.unsqueeze(x1r_laststep, dim=0)
        # print('x1r last',x1r_laststep_2.shape)

        # 'calzulate RNN attention, direction 1'
        # allpixels1 = x1r[:,1,:]
        # allpixels1 = self.linear(allpixels1)
        # print('allpixels1', allpixels1.shape)
        # pairdistance = nn.PairwiseDistance(p=2)
        # x1r_laststep_2 = self.linear(x1r_laststep_2)
        # output1 = pairdistance(allpixels1,x1r_laststep_2)
        # output1 = self.softmax(output1)
        # output1 = output1.unsqueeze(0)
        #
        # output1_1 = torch.matmul(allpixels1,x1r_laststep_2)
        # output1_1 =self.softmax(output1_1)
        # output1_1 = output1_1.unsqueeze(0)
        # # print('output12',output12)
        # # plt.plot(output1_1.cpu().detach().numpy(), linewidth = 2, marker = 'o')
        # # plt.show()
        # # a1 = sns.heatmap(data=output1.cpu().detach().numpy(), cmap="Blues", linewidths=0.1)
        # # a1.set_title('direction1')
        # # plt.show()
        '----------------------------------------------------------------------------------------------------'
        h0_x2r = torch.zeros(1, x2r.size(1), 64).to(device='cuda')
        c0_x2r = torch.zeros(1, x2r.size(1), 64).to(device="cuda")
        x2r = self.lstm_2_2(x2r)[0]
        # x2r = self.gru_2_2(x2r)[0] #把x1r经过RNN的值，作为x2r的输入
        x2r_laststep = x2r[-1]
        # x2r_laststep_2 = x2r[-1, 1, :]
        x2r_laststep = self.relu(x2r_laststep)
        x2r_laststep = torch.unsqueeze(x2r_laststep, dim=0)

        # 'calzulate RNN attention, direction 2'
        # allpixels2 = x2r[:,1,:]
        # allpixels2 = self.linear(allpixels2)
        # x2r_laststep_2 = self.linear(x2r_laststep_2)
        # output2 = pairdistance(allpixels2,x2r_laststep_2)
        # output2 = self.softmax(output2)
        # output2 = output2.unsqueeze(0)
        #
        # output2_2 = torch.matmul(allpixels2,x2r_laststep_2)
        # output2_2 =self.softmax(output2_2)
        # output2_2 = output2_2.unsqueeze(0)
        # # plt.plot(output2[0,:].cpu().detach().numpy(), linewidth =1.5)
        # # plt.show()
        # # a1 = sns.heatmap(data=output2.cpu().detach().numpy(), cmap="Blues", linewidths=0.1)
        # # a1.set_title('direction2')
        # # plt.show()
        '----------------------------------------------------------------------------------------------------------'

        h0_x3r = torch.zeros(1, x3r.size(1), 64).to(device='cuda')
        c0_x3r = torch.zeros(1, x3r.size(1), 64).to(device="cuda")
        x3r = self.lstm_2_3(x3r)[0]
        # x3r = self.gru_2_3(x3r)[0]
        x3r_laststep = x3r[-1]
        # x3r_laststep_2 = x3r[-1, 1, :]
        x3r_laststep = self.relu(x3r_laststep)
        x3r_laststep = torch.unsqueeze(x3r_laststep, dim=0)

        # 'calzulate RNN attention, direction 3'
        # allpixels3 = x3r[:, 1, :]
        # allpixels3 =self.linear(allpixels3)
        # x3r_laststep_2 = self.linear(x3r_laststep_2)
        # output3 = pairdistance(allpixels3, x3r_laststep_2)
        # output3 = self.softmax(output3)
        # output3 = output3.unsqueeze(0)
        # print('output3', output3)
        #
        # output3_3 = torch.matmul(allpixels3,x3r_laststep_2)
        # output3_3 =self.softmax(output3_3)
        # output3_3 = output3_3.unsqueeze(0)
        # # plt.plot(output3[0, :].cpu().detach().numpy(), linewidth=1.5)
        # # plt.show()
        # # a1 = sns.heatmap(data=output3.cpu().detach().numpy(), cmap="Blues", linewidths=0.1)
        # # a1.set_title('direction3')
        # # plt.show()
        '------------------------------------------------------------------------------------------------------------'

        h0_x4r = torch.zeros(1, x4r.size(1), 64).to(device='cuda')
        c0_x4r = torch.zeros(1, x4r.size(1), 64).to(device="cuda")
        x4r = self.lstm_2_4(x4r)[0]
        # x4r = self.gru_2_4(x4r)[0]
        x4r_laststep = x4r[-1]
        # x4r_laststep_2 = x4r[-1, 1, :]
        x4r_laststep = self.relu(x4r_laststep)
        x4r_laststep = torch.unsqueeze(x4r_laststep, dim=0)

        # 'calzulate RNN attention, direction 4'
        # allpixels4 = x4r[:, 1, :]
        # allpixels4 = self.linear(allpixels4)
        # x4r_laststep_2 = self.linear(x4r_laststep_2)
        # output4 = pairdistance(allpixels4, x4r_laststep_2)
        # output4 = self.softmax(output4)
        # output4 = output4.unsqueeze(0)
        #
        # output4_4 = torch.matmul(allpixels4,x4r_laststep_2)
        # output4_4 =self.softmax(output4_4)
        # output4_4 = output4_4.unsqueeze(0)
        # # plt.plot(output4[0, :].cpu().detach().numpy(), linewidth=1.5)
        # # plt.show()
        # # a1 = sns.heatmap(data=output4.cpu().detach().numpy(), cmap="Blues", linewidths=0.1)
        # # a1.set_title('direction4')
        # # plt.show()
        '------------------------------------------------------------------------------------------------------------'

        h0_x5r = torch.zeros(1, x5r.size(1), 64).to(device='cuda')
        c0_x5r = torch.zeros(1, x5r.size(1), 64).to(device="cuda")
        x5r = self.lstm_2_5(x5r)[0]
        # x5r = self.gru_2_5(x5r)[0]
        x5r_laststep = x5r[-1]
        # x5r_laststep_2 = x5r[-1, 1, :]
        x5r_laststep = self.relu(x5r_laststep)
        x5r_laststep = torch.unsqueeze(x5r_laststep, dim=0)

        # 'calzulate RNN attention, direction 5'
        # allpixels5 = x5r[:, 1, :]
        # allpixels5 = self.linear(allpixels5)
        # x5r_laststep_2 = self.linear(x5r_laststep_2)
        # output5 = pairdistance(allpixels5, x5r_laststep_2)
        # output5 = self.softmax(output5)
        # output5 = output5.unsqueeze(0)
        #
        # output5_5 = torch.matmul(allpixels5,x5r_laststep_2)
        # output5_5 =self.softmax(output5_5)
        # output5_5 = output5_5.unsqueeze(0)
        # # plt.plot(output5[0, :].cpu().detach().numpy(), linewidth=1.5)
        # # plt.show()
        # # a1 = sns.heatmap(data=output5.cpu().detach().numpy(), cmap="Blues", linewidths=0.1)
        # # a1.set_title('direction5')
        # # plt.show()
        '------------------------------------------------------------------------------------------------------------'

        h0_x6r = torch.zeros(1, x6r.size(1), 64).to(device='cuda')
        c0_x6r = torch.zeros(1, x6r.size(1), 64).to(device="cuda")
        x6r = self.lstm_2_6(x6r)[0]
        # x6r = self.gru_2_6(x6r)[0]
        x6r_laststep = x6r[-1]
        # x6r_laststep_2 = x6r[-1, 1, :]
        x6r_laststep = self.relu(x6r_laststep)
        x6r_laststep = torch.unsqueeze(x6r_laststep, dim=0)

        # 'calzulate RNN attention, direction 6'
        # allpixels6 = x6r[:, 1, :]
        # allpixels6 = self.linear(allpixels6)
        # x6r_laststep_2 = self.linear(x6r_laststep_2)
        # output6 = pairdistance(allpixels6, x6r_laststep_2)
        # output6 = self.softmax(output6)
        # output6 = output6.unsqueeze(0)
        #
        # output6_6 = torch.matmul(allpixels6,x6r_laststep_2)
        # output6_6 =self.softmax(output6_6)
        # output6_6 = output6_6.unsqueeze(0)
        # # plt.plot(output6[0, :].cpu().detach().numpy(), linewidth=1.5)
        # # plt.show()
        # # a1 = sns.heatmap(data=output6.cpu().detach().numpy(), cmap="Blues", linewidths=0.1)
        # # a1.set_title('direction6')
        # # plt.show()
        '------------------------------------------------------------------------------------------------------------'

        h0_x7r = torch.zeros(1, x7r.size(1), 64).to(device='cuda')
        c0_x7r = torch.zeros(1, x7r.size(1), 64).to(device="cuda")
        x7r = self.lstm_2_7(x7r)[0]
        # x7r = self.gru_2_7(x7r)[0]
        x7r_laststep = x7r[-1]
        # x7r_laststep_2 = x7r[-1, 1, :]
        x7r_laststep = self.relu(x7r_laststep)
        x7r_laststep = torch.unsqueeze(x7r_laststep, dim=0)
        #
        # 'calzulate RNN attention, direction 7'
        # allpixels7 = x7r[:, 1, :]
        # allpixels7 = self.linear(allpixels7)
        # x7r_laststep_2 = self.linear(x7r_laststep_2)
        # output7 = pairdistance(allpixels7, x7r_laststep_2)
        # output7 = self.softmax(output7)
        # output7 = output7.unsqueeze(0)
        #
        # output7_7 = torch.matmul(allpixels7,x7r_laststep_2)
        # output7_7 =self.softmax(output7_7)
        # output7_7 = output7_7.unsqueeze(0)
        # # plt.plot(output7[0, :].cpu().detach().numpy(), linewidth=1.5)
        # # plt.show()
        # # a1 = sns.heatmap(data=output7.cpu().detach().numpy(), cmap="Blues", linewidths=0.1)
        # # a1.set_title('direction7')
        # # plt.show()
        '------------------------------------------------------------------------------------------------------------'

        h0_x8r = torch.zeros(1, x8r.size(1), 64).to(device='cuda')
        c0_x8r = torch.zeros(1, x8r.size(1), 64).to(device="cuda")
        x8r = self.lstm_2_8(x8r)[0]
        # x8r = self.gru_2_8(x8r)[0]
        x8r_laststep = x8r[-1]
        # x8r_laststep_2 = x8r[-1, 1, :]
        x8r_laststep = self.relu(x8r_laststep)
        x8r_laststep = torch.unsqueeze(x8r_laststep, dim=0)
        print('x8r_last',x8r_laststep.shape)

        # 'calzulate RNN attention, direction 8'
        # allpixels8 = x8r[:, 1, :]
        # allpixels8 = self.linear(allpixels8)
        # x8r_laststep_2 = self.linear(x8r_laststep_2)
        # output8 = pairdistance(allpixels8, x8r_laststep_2)
        # output8 = self.softmax(output8)
        # output8 = output8.unsqueeze(0)
        #
        # output8_8 = torch.matmul(allpixels8,x8r_laststep_2)
        # output8_8 =self.softmax(output8_8)
        # output8_8 = output8_8.unsqueeze(0)
        # # plt.plot(output8[0, :].cpu().detach().numpy(), linewidth=1.5)
        # # plt.show()
        # # a1 = sns.heatmap(data=output8.cpu().detach().numpy(), cmap="Blues", linewidths=0.1)
        # # a1.set_title('direction8')
        # # plt.show()
        '----show attetntion function------------------------------------------------------'
        def showattention(inputseq):
            allpixel = inputseq[:, 1, :]
            linear1 = nn.Linear(allpixel.size(1),allpixel.size(1)).to( device='cuda')
            allpixel = linear1(allpixel)

            # centralstep = allpixel[12,:]
            laststep = inputseq[-1, 1, :]
            laststep = linear1(laststep)

            output = torch.matmul(allpixel, laststep.transpose(0,-1))

            pairdis = nn.PairwiseDistance()
            cos = nn.CosineSimilarity(dim=-1)

            output_pair = pairdis(allpixel,laststep) * -1
            # output_pair = cos(allpixel, laststep)

            softmax = nn.Softmax()
            output = softmax(output)
            output_pair = softmax(output_pair)
            output = output.unsqueeze(0)
            output_pair = output_pair.unsqueeze(0)
            print('cos',output_pair.shape)
            return output,output_pair
        '------------------------------------------------------------------------------------'
        # output1_1,output1_1_cos = showattention(x1r)
        # # sns.lineplot(data=output1_1_cos.transpose(1,0).cpu().detach().numpy(), markers=True, lw=2)
        # output2_2, output2_2_cos = showattention(x2r)
        # # sns.lineplot(data=output2_2_cos.transpose(1,0).cpu().detach().numpy(), markers=True, lw=2)
        # output3_3,output3_3_cos = showattention(x3r)
        # # sns.lineplot(data=output3_3_cos.transpose(1,0).cpu().detach().numpy(), markers=True, lw=2)
        # output4_4,output4_4_cos = showattention(x4r)
        # # sns.lineplot(data=output4_4_cos.transpose(1,0).cpu().detach().numpy(), markers=True, lw=2)
        # output5_5,output5_5_cos = showattention(x5r)
        # # sns.lineplot(data=output5_5_cos.transpose(1,0).cpu().detach().numpy(), markers=True, lw=2)
        # output6_6,output6_6_cos = showattention(x6r)
        # # sns.lineplot(data=output6_6_cos.transpose(1,0).cpu().detach().numpy(), markers=True, lw=2)
        # output7_7,output7_7_cos = showattention(x7r)
        # # sns.lineplot(data=output7_7_cos.transpose(1,0).cpu().detach().numpy(), markers=True, lw=2)
        # output8_8,output8_8_cos = showattention(x8r)
        # # sns.lineplot(data=output8_8_cos.transpose(1,0).cpu().detach().numpy(), markers=True, lw=2)
        # # plt.show()
        '------------------------------------------------------------------------------------------------------------'
        # outputall = torch.cat([output1_1_cos, output2_2_cos, output3_3_cos, output4_4_cos, output5_5_cos, output6_6_cos, output7_7_cos, output8_8_cos],dim=0)
        # sns.lineplot(data=outputall.transpose(1,0).cpu().detach().numpy(), markers=True, lw=2)
        # plt.show()
        # all = sns.heatmap(data=outputall.cpu().detach().numpy(), cmap="mako", linewidths=0.05, square=True)
        # all.set_title('all')
        # plt.show()

        '第一种合并'
        # # x = x1r_laststep + x2r_laststep + x3r_laststep + x4r_laststep + x5r_laststep + x6r_laststep + x7r_laststep + x8r_laststep
        # # x = x.squeeze(0)
        # x = x1r + x2r + x3r+ x4r + x5r + x6r + x7r + x8r
        # print('x.shape',x.shape)
        # x = x.permute(1, 2, 0).contiguous()
        # # x = x.permute(0, 1).contiguous()
        # x = x.view(x.size(0), -1)
        # x = self.lstm_bn_1_2(x)
        # x = self.prelu(x)
        # # x = self.dropout(x)
        # x = self.lstm_fc_1_2(x)

        # '第二种合并'
        # x = torch.cat([x1r_laststep, x2r_laststep, x3r_laststep, x4r_laststep, x5r_laststep, x6r_laststep, x7r_laststep,
        #                x8r_laststep], dim=2)
        # # x = torch.cat([x1r,x2r,x3r,x4r,x5r,x6r,x7r,x8r],dim=2)
        # print('x.shape',x.shape)
        # x = x.squeeze(0)
        # x = x.permute(0, 1).contiguous()
        # # x = x.permute(1, 2, 0).contiguous()
        # x = x.view(x.size(0), -1)
        # x = self.lstm_bn_2(x)
        # # x = self.lstm_bn_2_2(x)
        # x = self.prelu(x)
        # x = self.dropout(x)
        # x = self.lstm_fc_2(x)
        # # x = self.lstm_fc_2_2(x)


        "第三种合并"
        x_strategy_1 = torch.cat([x8r_laststep,x7r_laststep,x6r_laststep,x5r_laststep,x4r_laststep,x3r_laststep,x2r_laststep,x1r_laststep],dim=0)
        x_strategy_1 = rearrange(x_strategy_1, 'n b d -> b n d')
        x_strategy_1 = self.scan_order(x_strategy_1)
        # print('x_strategy_1', x_strategy_1.shape) #(8 , batch, 64)
        # x_strategy_1 = x_strategy_1.permute(1, 0, 2).contiguous()#(100,64,8)
        h0_last = torch.zeros(1, x_strategy_1.size(1), 64).to(device='cuda')
        c0_last = torch.zeros(1, x_strategy_1.size(1), 64).to(device="cuda")
        x_strategy_1 = self.lstm_stra_1(x_strategy_1)
        # x_strategy_1 = self.gru_4(x_strategy_1)[0]
        x_strategy_1_laststep = x_strategy_1[-1]
        # x_strategy_1_laststep_2 = x_strategy_1[-1, 1, :]
        # x_strategy_1_laststep = x_strategy_1.permute(1, 2, 0).contiguous()
        # print('x_strategy_1_laststep',x_strategy_1_laststep.shape)
        # np.save('x_strategy_1_laststep', x_strategy_1_laststep.cpu().detach().numpy(), allow_pickle=True)
        '------------------------------------------'
        'calzulate RNN attention for 8 directions'

        '-------------------------------------------------------------------------------------'
        x_strategy_1_laststep = x_strategy_1_laststep.permute(0, 1).contiguous()
        x_strategy_1_laststep = x_strategy_1_laststep.view(x_strategy_1_laststep.size(0), -1)
        # x_strategy_1_laststep = self.gru_bn_4(x_strategy_1_laststep)
        x_strategy_1_laststep = self.gru_bn_laststep(x_strategy_1_laststep)
        x_strategy_1_laststep = self.prelu(x_strategy_1_laststep)
        x_strategy_1_laststep = self.dropout(x_strategy_1_laststep)
        # x_strategy_1_laststep = self.fc_4(x_strategy_1_laststep)
        x_strategy_1_laststep = self.fc_laststep(x_strategy_1_laststep)

        # var2 = torch.var(x_strategy_1_laststep)
        # print('var2:', var2)

        x = x_strategy_1_laststep
        # 下面改变输入值，确定使用哪个方向


        # x1r = x1r.permute(1, 2, 0).contiguous()
        # x2r = x2r.permute(1, 2, 0).contiguous()
        # x3r = x3r.permute(1, 2, 0).contiguous()
        # x4r = x4r.permute(1, 2, 0).contiguous()
        # x5r = x5r.permute(1, 2, 0).contiguous()
        # x6r = x6r.permute(1, 2, 0).contiguous()
        # x7r = x7r.permute(1, 2, 0).contiguous()
        # x8r = x8r.permute(1, 2, 0).contiguous()
        # x_strategy_1 = x_strategy_1.permute(1,2,0).contiguous()


        # x1r = x1r.view(x1r.size(0), -1)
        # x2r = x2r.view(x2r.size(0), -1)
        # x3r = x3r.view(x3r.size(0), -1)
        # x4r = x4r.view(x4r.size(0), -1)
        # x5r = x5r.view(x5r.size(0), -1)
        # x6r = x6r.view(x6r.size(0), -1)
        # x7r = x7r.view(x7r.size(0), -1)
        # x8r = x8r.view(x8r.size(0), -1)
        # x_strategy_1 = x_strategy_1.view(x_strategy_1.size(0),-1)


        # x1r = self.gru_bn_3(x1r)
        # x2r = self.gru_bn_3(x2r)
        # x3r = self.gru_bn_3(x3r)
        # x4r = self.gru_bn_3(x4r)
        # x5r = self.gru_bn_3(x5r)
        # x6r = self.gru_bn_3(x6r)
        # x7r = self.gru_bn_3(x7r)
        # x8r = self.gru_bn_3(x8r)
        # x_strategy_1 = self.gru_bn_4(x_strategy_1)


        # x1r = self.tanh(x1r)
        # x2r = self.tanh(x2r)
        # x3r = self.tanh(x3r)
        # x4r = self.tanh(x4r)
        # x5r = self.tanh(x5r)
        # x6r = self.tanh(x6r)
        # x7r = self.tanh(x7r)
        # x8r = self.tanh(x8r)
        #
        # x1r = self.dropout(x1r)
        #
        # x2r = self.dropout(x2r)
        #
        # x3r = self.dropout(x3r)
        #
        #
        # x4r = self.dropout(x4r)
        #
        #
        # x5r = self.dropout(x5r)
        #
        #
        # x6r = self.dropout(x6r)
        #
        #
        # x7r = self.dropout(x7r)
        #
        #
        # x8r = self.dropout(x8r)
        # x_strategy_1 = self.dropout(x_strategy_1)
        # x_strategy_1_laststep = self.dropout(x_strategy_1_laststep)

        # plt.subplot(3, 3, 1).set_title('Spectral signatures in a patch')
        # plt.xlabel('Band Numbers', fontdict={'size': 15}, fontweight='bold')
        # plt.ylabel('Spectral Values Values', fontdict={'size': 15}, fontweight='bold')
        # plt.plot(direction_1[0, :, :].cpu().detach().numpy())

        # x1r = self.fc_3(x1r)
        # plt.subplot(3, 3, 2)
        # plt.xlabel('Feature size', fontdict={'size': 15}, fontweight='bold')
        # plt.ylabel('Feature Values', fontdict={'size': 15}, fontweight='bold')
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.plot(x1r[0, :].cpu().detach().numpy())

        # x2r = self.fc_3(x2r)
        # plt.subplot(3, 3, 3)
        # plt.xlabel('Feature size', fontdict={'size': 15}, fontweight='bold')
        # plt.ylabel('Feature Values', fontdict={'size': 15}, fontweight='bold')
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.plot(x2r[0, :].cpu().detach().numpy())

        # x3r = self.fc_3(x3r)
        # plt.subplot(3, 3, 4)
        # plt.xlabel('Feature size', fontdict={'size': 15}, fontweight='bold')
        # plt.ylabel('Feature Values', fontdict={'size': 15}, fontweight='bold')
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.plot(x3r[0, :].cpu().detach().numpy())

        # x4r = self.fc_3(x4r)
        # plt.subplot(3, 3, 5)
        # plt.xlabel('Feature size', fontdict={'size': 15}, fontweight='bold')
        # plt.ylabel('Feature Values', fontdict={'size': 15}, fontweight='bold')
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.plot(x4r[0, :].cpu().detach().numpy())

        # x5r = self.fc_3(x5r)
        # plt.subplot(3, 3, 6)
        # plt.xlabel('Feature size', fontdict={'size': 15}, fontweight='bold')
        # plt.ylabel('Feature Values', fontdict={'size': 15}, fontweight='bold')
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.plot(x5r[0, :].cpu().detach().numpy())

        # x6r = self.fc_3(x6r)
        # plt.subplot(3, 3, 7)
        # plt.xlabel('Feature size', fontdict={'size': 15}, fontweight='bold')
        # plt.ylabel('Feature Values', fontdict={'size': 15}, fontweight='bold')
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.plot(x6r[0, :].cpu().detach().numpy())

        # x7r = self.fc_3(x7r)
        # plt.subplot(3, 3, 8)
        # plt.xlabel('Feature size', fontdict={'size': 15}, fontweight='bold')
        # plt.ylabel('Feature Values', fontdict={'size': 15}, fontweight='bold')
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.plot(x7r[0, :].cpu().detach().numpy())

        # x8r = self.fc_3(x8r)
        # plt.subplot(3, 3, 9)
        # plt.xlabel('Feature Values', fontdict={'size': 15}, fontweight='bold')
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.plot(x8r[0, :].cpu().detach().numpy())
        # plt.show()

        # x_strategy_1 = self.fc_4(x_strategy_1)'Feature size', fontdict={'size': 15}, fontweight='bold')
        # plt.ylabel('Feature Values', fontdict={'size': 15}, fontweight='bold')
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.plot(x8r[0, :].cpu().detach().numpy())
        # plt.show()

        # x_strategy_1 = self.fc_4(x_strategy_1)

        return x

class Order_Attention(nn.Module):
    def __init__(self, hidden_size, attention_size):
        super(Order_Attention, self).__init__()
        self.w_omega = nn.Parameter(torch.randn(hidden_size, attention_size)) # [hidden_size, attention_size]
        self.b_omega = nn.Parameter(torch.randn(attention_size)) # [attention_size]
        self.u_omega = nn.Parameter(torch.randn(attention_size)) # [attention_size]

    def forward(self, inputs):
        # inputs: [seq_len, batch_size, hidden_size]
        inputs = inputs.permute(1, 0, 2) # inputs: [batch_size, seq_len, hidden_size]
        v = torch.tanh(torch.matmul(inputs, self.w_omega) + self.b_omega) # v: [batch_size, seq_len, attention_size]
        vu = torch.matmul(v, self.u_omega) # vu: [batch_size, seq_len]
        alphas = F.softmax(vu, dim=1) # alphas: [batch_size, seq_len]
        output = inputs * alphas.unsqueeze(-1) # output: [batch_size, STEP, hidden_size]
        return output, alphas # output: [batch_size, hidden_size], alphas: [batch_size, seq_len]


class ARNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, attention_size):
        super(ARNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, bidirectional=False) # output: [seq_len, batch_size, 2*hidden_size]
        self.attention = Order_Attention(hidden_size, attention_size)
        self.fc = nn.Linear(hidden_size, output_size) # output: [batch_size, output_size]

    def forward(self, x):
        # x: [seq_len, batch_size, input_size]
        outputs, _ = self.gru(x) # outputs: [seq_len, batch_size, 2*hidden_size]
        a_output, alphas = self.attention(outputs) # a_output: [batch_size, 2*hidden_size], alphas: [batch_size, seq_len]
        outputs = self.fc(a_output) # outputs: [batch_size, output_size]
        return outputs # outputs: [batch_size, output_size]

class zhouEightDRNN_kamata_Transformer(nn.Module):

    @staticmethod
    def weight_init(m):
        # All weight matrices in our RNN and bias vectors are initialized with a uniform distribution, and the values of these weight matrices and bias vectors are initialized in the range [−0.1,0.1]
        if isinstance(m, (nn.Linear, nn.GRU, nn.Conv3d,nn.LSTM, nn.TransformerEncoderLayer, nn.TransformerEncoder)):
            init.kaiming_uniform_(m.weight.data)
            init.kaiming_uniform_(m.bias.data)

    def __init__(self, input_channels, n_classes, patch_size=5, embed_dim = 64):
        # The proposed network model uses a single recurrent layer that adopts our modified GRUs of size 64 with sigmoid gate activation and PRetanh activation functions for hidden representations
        super(zhouEightDRNN_kamata_Transformer, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        'do not use self.n_classes = n_classes'
        self.lstm_trans = nn.LSTM(4, 65 * 4, 1, bidirectional=False)
        self.lstm_2_1 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.lstm_2_2 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.lstm_2_3 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.lstm_2_4 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.lstm_2_5 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.lstm_2_6 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.lstm_2_7 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.lstm_2_8 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.cls_token_1 = nn.Parameter(torch.randn(1, 25, 1))
        self.cls_token_2 = nn.Parameter(torch.randn(1, 25, 1))
        self.cls_token_3 = nn.Parameter(torch.randn(1, 25, 1))
        self.cls_token_4 = nn.Parameter(torch.randn(1, 25, 1))
        self.cls_token_5 = nn.Parameter(torch.randn(1, 25, 1))
        self.cls_token_6 = nn.Parameter(torch.randn(1, 25, 1))
        self.cls_token_7 = nn.Parameter(torch.randn(1, 25, 1))
        self.cls_token_8 = nn.Parameter(torch.randn(1, 25, 1))

        self.pos_embedding_1 = nn.Parameter(torch.randn(1, 25, embed_dim))
        self.pos_embedding_2 = nn.Parameter(torch.randn(1, 25 +1, embed_dim))
        self.pos_embedding_3 = nn.Parameter(torch.randn(1, 25 +1, embed_dim))
        self.pos_embedding_4 = nn.Parameter(torch.randn(1, 25 +1, embed_dim))
        self.pos_embedding_5 = nn.Parameter(torch.randn(1, 25 +1, embed_dim))
        self.pos_embedding_6 = nn.Parameter(torch.randn(1, 25 +1, embed_dim))
        self.pos_embedding_7 = nn.Parameter(torch.randn(1, 25 +1, embed_dim))
        self.pos_embedding_8 = nn.Parameter(torch.randn(1, 25 +1, embed_dim))
        self.pos_embedding_conca_scheme1 = nn.Parameter(torch.randn(1, 8, embed_dim))
        self.pos_embedding_conca_scheme2 = nn.Parameter(torch.randn(1, 4 + 1, embed_dim * 4))
        self.cls_token_FLC = nn.Parameter(torch.randn(1, 1, embed_dim * 4))
        self.lstm_4 = nn.LSTM(patch_size ** 2, 64, 1)
        self.transformer = nn.Transformer(nhead=16, num_encoder_layers=12)
        self.conv2d_1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=2, stride=1)
        self.dpe = nn.Conv2d(in_channels=input_channels,out_channels=1, kernel_size=1)
        self.dpe_2 = nn.Conv2d(in_channels=input_channels, out_channels=25, kernel_size=1)
        self.depth_conv_1 = nn.Conv2d(in_channels=input_channels,out_channels=input_channels,kernel_size=3, padding=1, groups=input_channels)
        self.point_conv_1 = nn.Conv2d(in_channels=input_channels,out_channels=input_channels,kernel_size=1)
        self.depth_conv_2 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3)
        self.point_conv_2 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3)
        self.conv2d_2 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=2, stride=1)
        self.conv2d_3 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, stride=1)
        self.conv2d_4 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1)
        self.gru_bn = nn.BatchNorm1d(patch_size ** 2 * input_channels)
        self.lstm_bn_1 = nn.BatchNorm1d((64)*1)
        self.lstm_bn_1_2 = nn.BatchNorm1d((64) * (patch_size)**2)
        self.lstm_bn_2 = nn.BatchNorm1d((64)*8)
        self.lstm_bn_2_2 = nn.BatchNorm1d((64) * 8 * patch_size**2)
        self.gru_bn_2 = nn.BatchNorm1d((patch_size ** 2) * (patch_size**2))
        self.gru_bn_3 = nn.BatchNorm1d((patch_size ** 2) * (patch_size ** 2) * 2)
        self.gru_bn_4 = nn.BatchNorm1d(8 * 64)
        self.gru_bn_laststep = nn.BatchNorm1d(64)
        self.transformer_bn_scheme1 = nn.BatchNorm1d((64) * 8 )
        self.transformer_bn_scheme2 = nn.BatchNorm1d(embed_dim * 2)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(67600, n_classes)
        self.bn = nn.BatchNorm1d(67600)
        self.lstm_fc_1 = nn.Linear(64 * 1, n_classes)
        self.lstm_fc_1_2 = nn.Linear(64 * (patch_size**2), n_classes)
        self.lstm_fc_2 = nn.Linear(64*8,n_classes)
        self.lstm_fc_2_2 = nn.Linear(64 * 8 * patch_size**2, n_classes)
        self.fc_2 = nn.Linear((patch_size ** 2) * (patch_size**2), n_classes)
        self.fc_3 = nn.Linear((patch_size ** 2) * (patch_size ** 2) * 2, n_classes)
        self.fc_4 = nn.Linear(8 * 64, n_classes)
        self.layernorm = nn.LayerNorm(embed_dim)
        self.denselayer = nn.Linear(64,64)
        self.denselayer_scheme1 = nn.Linear(embed_dim,embed_dim)
        self.denselayer_scheme2 = nn.Linear(embed_dim * 4, embed_dim * 2)
        self.fc_laststep = nn.Linear(64, n_classes)
        self.fc_transformer_scheme1 = nn.Linear((64) *8,n_classes)
        self.fc_transformer_scheme2 = nn.Linear(embed_dim * 2 , n_classes)
        self.linearembedding = nn.Linear(in_features=input_channels,out_features=embed_dim)
        self.softmax = nn.Softmax(dim=1)
        self.aux_loss_weight = 1

    def forward(self, x): #初始是第1方向
        print('x.shape1',x.shape)
        x = x.squeeze(1) #b,c,h w
        vis.images(x[1, 0 ,:,:])
        x1 = x #用于保存原数据
        # x2 = x #用于生成dpe
        # dpe = self.dpe(x2)
        # print(dpe.dtype)
        # print('dpe',dpe) #(b 1 5 5)
        #
        #
        # #试一试one-hot
        # # dpe_onehot = one_hot_extend(dpe)
        # #
        # # onehotencoder = OneHotEncoder(categories='auto')
        # # for i in range(dpe_onehot.size(0)):
        # #     i_th = dpe_onehot[i,:]
        # #     i_th = i_th.unsqueeze(0)
        # #     onehotoutput = onehotencoder.fit_transform(i_th.cpu().detach().numpy())
        # #     print(onehotoutput.toarray())
        # # i = i + 1
        # #
        # #
        #
        #
        #
        # #multi-dpe (把dpe也换成multiscanning)
        # #生成1和7
        # dpe1_0 = dpe[:, :, 0, :]
        # dpe1_1 = dpe[:, :, 1, :]
        # dpe1_2 = dpe[:, :, 2, :]
        # dpe1_3 = dpe[:, :, 3, :]
        # dpe1_4 = dpe[:, :, 4, :]
        # dpe1_1f = torch.flip(dpe1_1, [2])
        # dpe1_3f = torch.flip(dpe1_3, [2])
        # # direction_1 = torch.cat([x1_0, x1_1f, x1_2], dim=2)
        # dpe_1 = torch.cat([dpe1_0, dpe1_1f, dpe1_2,dpe1_3f,dpe1_4], dim=2)
        # print('dpe_1',dpe_1.shape) #（100, 1 ,25）(b c x)
        # dpe_7 = torch.flip(dpe_1, [2])
        # dpe_1 = repeat(dpe_1,'b 1 x -> b x 25')
        # dpe_7 = repeat(dpe_7,'b 1 x -> b x 25')
        # print('dpe_1', dpe_1.shape)  # （100, 1 ,25）(b c x)
        # # plt.subplot(241)
        # # plt.imshow(dpe_1[1, :, :].cpu().detach().numpy())
        # # plt.subplot(247)
        # # plt.imshow(dpe_7[1, :, :].cpu().detach().numpy())
        # # plt.show()
        #
        #
        # # 生成第2和8
        # dpe2_0 = dpe[:, :, :, 0]
        # dpe2_1 = dpe[:, :, :, 1]
        # dpe2_2 = dpe[:, :, :, 2]
        # dpe2_3 = dpe[:, :, :, 3]
        # dpe2_4 = dpe[:, :, :, 4]
        # dpe2_1f = torch.flip(dpe2_1, [2])
        # dpe2_3f = torch.flip(dpe2_3, [2])
        # # direction_2 = torch.cat([x2_0, x2_1f, x2_2], dim=2)
        # dpe_2 = torch.cat([dpe2_0, dpe2_1f, dpe2_2, dpe2_3f, dpe2_4], dim=2)
        # dpe_8 = torch.flip(dpe_2, [2])
        # dpe_2 = repeat(dpe_2, 'b 1 x -> b x 25')
        # dpe_8 = repeat(dpe_8, 'b 1 x -> b x 25')
        # # plt.subplot(242)
        # # plt.imshow(dpe_2[1, :, :].cpu().detach().numpy())
        # # plt.subplot(248)
        # # plt.imshow(dpe_8[1, :, :].cpu().detach().numpy())
        #
        # # 生成3和5
        # dpe3_0 = dpe[:, :, 0, :]
        # dpe3_1 = dpe[:, :, 1, :]
        # dpe3_2 = dpe[:, :, 2, :]
        # dpe3_3 = dpe[:, :, 3, :]
        # dpe3_4 = dpe[:, :, 4, :]
        # dpe3_0f = torch.flip(dpe3_0, [2])
        # dpe3_2f = torch.flip(dpe3_2, [2])
        # dpe3_4f = torch.flip(dpe3_4, [2])
        # # direction_3 = torch.cat([x3_0f, x3_1, x3_2f], dim=2)
        # dpe_3 = torch.cat([dpe3_0f, dpe3_1, dpe3_2f, dpe3_3, dpe3_4f], dim=2)
        # dpe_5 = torch.flip(dpe_3, [2])
        # dpe_3 = repeat(dpe_3, 'b 1 x -> b x 25')
        # dpe_5 = repeat(dpe_5, 'b 1 x -> b x 25')
        # # plt.subplot(243)
        # # plt.imshow(dpe_3[1, :, :].cpu().detach().numpy())
        # # plt.subplot(245)
        # # plt.imshow(dpe_5[1, :, :].cpu().detach().numpy())
        #
        # # 生成4和6
        # dpe4_0 = dpe[:, :, :, 0]
        # dpe4_1 = dpe[:, :, :, 1]
        # dpe4_2 = dpe[:, :, :, 2]
        # dpe4_3 = dpe[:, :, :, 3]
        # dpe4_4 = dpe[:, :, :, 4]
        # dpe4_1f = torch.flip(dpe4_1, [2])
        # dpe4_3f = torch.flip(dpe4_3, [2])
        # # direction_4 = torch.cat([x4_2, x4_1f, x4_0], dim=2)
        # dpe_4 = torch.cat([dpe4_4, dpe4_3f, dpe4_2, dpe4_1f, dpe4_0], dim=2)
        # # print('d4', direction_4.shape)
        # dpe_6 = torch.flip(dpe_4, [2])
        # dpe_4 = repeat(dpe_4, 'b 1 x -> b x 25')
        # dpe_6 = repeat(dpe_6, 'b 1 x -> b x 25')
        # # plt.subplot(244)
        # # plt.imshow(dpe_4[1, :, :].cpu().detach().numpy())
        # # plt.subplot(246)
        # # plt.imshow(dpe_6[1, :, :].cpu().detach().numpy())
        # # plt.show()
        #
        #
        # # dpe = reduce(dpe, 'b 1 h w -> b h w', reduction='mean')
        # # dpe = rearrange(dpe, 'b h w -> b (h w)')
        # # dpe = repeat(dpe, 'b x -> b x 25')
        # # print('dpe',dpe.shape)
        # # # dpe = self.sigmoid(dpe)
        # # plt.subplot(212)
        # # plt.imshow(dpe[1,:,:].cpu().detach().numpy())
        # # plt.show()
        # # print(dpe)
        # # xseq = rearrange(x,'b c h w -> b c (h w)')
        # # print(xseq.shape)
        #
        # # ResNet patch_size = 9 for SA PU
        # # x = self.conv2d_1(x)
        # # print('1', x.shape)
        # # x = self.relu(x)
        # # x = self.conv2d_2(x)
        # # print('2', x.shape)
        # # x_res = self.relu(x)
        # # x_res = self.conv2d_3(x_res)
        # # print('3', x.shape) #(ptach size = 6)
        # # x_res = self.relu(x_res)
        # # x_res_res = self.conv2d_4(x_res)
        # # x_res_res = self.relu(x_res_res)
        # # x = x_res + x_res_res
        # # print('4', x.shape)
        # #Depthwise separable convolution
        # # x1 = self.depth_conv_2(x)
        # # x1 = self.point_conv_2(x1)


        x = self.depth_conv_1(x)
        x = self.point_conv_1(x)
        x = self.relu(x)
        x = x1 + x
        x = rearrange(x,'b c h w -> b h w c')
        print('x.shape',x.shape)

        # x = self.relu(x)
        # x = x * x1

        # x = self.depth_conv_2(x)
        # x = self.point_conv_2(x)

        #生成第1和7
        # x1_0 = x[:, :, 0, :]
        # x1_1 = x[:, :, 1, :]
        # x1_2 = x[:, :, 2, :]
        # x1_3 = x[:, :, 3, :]
        # x1_4 = x[:, :, 4, :]
        # x1_1f = torch.flip(x1_1, [2])
        # x1_3f = torch.flip(x1_3, [2])
        # # direction_1 = torch.cat([x1_0, x1_1f, x1_2], dim=2)
        # direction_1 = torch.cat([x1_0, x1_1f, x1_2,x1_3f,x1_4], dim=2)
        direction_1 = multiscan(x, 1).cuda()
        direction_2 = multiscan(x, 2).cuda()
        direction_3 = multiscan(x, 3).cuda()
        direction_4 = multiscan(x, 4).cuda()
        direction_5 = multiscan(x, 5).cuda()
        direction_6 = multiscan(x, 6).cuda()
        direction_7 = multiscan(x, 7).cuda()
        direction_8 = multiscan(x, 8).cuda() #(b c s)


        #soft attention mask 速度会变慢
        direction_1_cdist = rearrange(direction_1, 'b c x -> b x c')
        dist = torch.cdist(direction_1_cdist,direction_1_cdist,p=2)
        print('dist',dist.shape)
        sns.heatmap(dist[1, :, :].cpu().detach().numpy(),cmap='Blues',linewidths=0.5,annot_kws={'fontweight':'bold'},square=True)
        # plt.colorbar()
        plt.show()
        weight = -dist
        weight = weight + 1
        sns.heatmap(weight[1, :, :].cpu().detach().numpy(),cmap='Blues',linewidths=0.5,annot_kws={'fontweight':'bold'},square=True)
        # plt.imshow(weight[1, :, :].cpu().detach().numpy(),cmap='blues')
        # plt.colorbar()
        plt.show()
        # print('d1',direction_1.shape)

        # direction_7 = torch.flip(direction_1,[2])
        #
        # #生成第2和8
        # x2_0 = x[:, :, :, 0]
        # x2_1 = x[:, :, :, 1]
        # x2_2 = x[:, :, :, 2]
        # x2_3 = x[:, :, :, 3]
        # x2_4 = x[:, :, :, 4]
        # x2_1f = torch.flip(x2_1, [2])
        # x2_3f = torch.flip(x2_3, [2])
        # # direction_2 = torch.cat([x2_0, x2_1f, x2_2], dim=2)
        # direction_2 = torch.cat([x2_0, x2_1f,x2_2,x2_3f,x2_4], dim=2)
        # direction_8 = torch.flip(direction_2, [2])

        # 生成3和5
        # x3_0 = x[:, :, 0, :]
        # x3_1 = x[:, :, 1, :]
        # x3_2 = x[:, :, 2, :]
        # x3_3 = x[:, :, 3, :]
        # x3_4 = x[:, :, 4, :]
        # x3_0f = torch.flip(x3_0, [2])
        # x3_2f = torch.flip(x3_2, [2])
        # x3_4f = torch.flip(x3_4, [2])
        # # direction_3 = torch.cat([x3_0f, x3_1, x3_2f], dim=2)
        # direction_3 = torch.cat([x3_0f, x3_1, x3_2f,x3_3,x3_4f], dim=2)
        # direction_5 = torch.flip(direction_3, [2])

        #生成4和6
        # x4_0 = x[:, :, :, 0]
        # x4_1 = x[:, :, :, 1]
        # x4_2 = x[:, :, :, 2]
        # x4_3 = x[:, :, :, 3]
        # x4_4 = x[:, :, :, 4]
        # x4_1f = torch.flip(x4_1, [2])
        # x4_3f = torch.flip(x4_3, [2])
        # # direction_4 = torch.cat([x4_2, x4_1f, x4_0], dim=2)
        # direction_4 = torch.cat([x4_4, x4_3f, x4_2, x4_1f, x4_0], dim=2)
        # # print('d4', direction_4.shape)
        # direction_6 = torch.flip(direction_4, [2])

        # plt.xlabel('Band Numbers', fontdict={'size': 35}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 35}, fontweight='bold')
        # plt.plot(direction_1[0, :, 0].cpu().detach().numpy(), label='index 0')
        # plt.plot(direction_1[0, :, 1].cpu().detach().numpy(), label='index 1')
        # plt.plot(direction_1[0, :, 2].cpu().detach().numpy(), label='index 2')
        # plt.plot(direction_1[0, :, 3].cpu().detach().numpy(), label='index 3')
        # plt.plot(direction_1[0, :, 4].cpu().detach().numpy(), label='index 4')
        # plt.plot(direction_1[0, :, 5].cpu().detach().numpy(), label='index 9')
        # plt.plot(direction_1[0, :, 6].cpu().detach().numpy(), label='index 8')
        # plt.plot(direction_1[0, :, 7].cpu().detach().numpy(), label='index 7')
        # plt.plot(direction_1[0, :, 8].cpu().detach().numpy(), label='index 6')
        # plt.plot(direction_1[0, :, 9].cpu().detach().numpy(), label='index 5')
        # plt.plot(direction_1[0, :, 10].cpu().detach().numpy(), label='index 10')
        # plt.plot(direction_1[0, :, 11].cpu().detach().numpy(), label='index 11')
        # plt.plot(direction_1[0, :, 12].cpu().detach().numpy(), label='index 12', linewidth=5, linestyle='-.', color = 'red' )
        # plt.plot(direction_1[0, :, 13].cpu().detach().numpy(), label='index 13')
        # plt.plot(direction_1[0, :, 14].cpu().detach().numpy(), label='index 14')
        # plt.plot(direction_1[0, :, 15].cpu().detach().numpy(), label='index 19')
        # plt.plot(direction_1[0, :, 16].cpu().detach().numpy(), label='index 18')
        # plt.plot(direction_1[0, :, 17].cpu().detach().numpy(), label='index 17')
        # plt.plot(direction_1[0, :, 18].cpu().detach().numpy(), label='index 16')
        # plt.plot(direction_1[0, :, 19].cpu().detach().numpy(), label='index 15')
        # plt.plot(direction_1[0, :, 20].cpu().detach().numpy(), label='index 20')
        # plt.plot(direction_1[0, :, 21].cpu().detach().numpy(), label='index 21')
        # plt.plot(direction_1[0, :, 22].cpu().detach().numpy(), label='index 22')
        # plt.plot(direction_1[0, :, 23].cpu().detach().numpy(), label='index 23')
        # plt.plot(direction_1[0, :, 24].cpu().detach().numpy(), label='index 24')
        # plt.legend(prop={'size': 15}, fontsize='large', loc=[1, 0])
        # plt.title('Spectral signatures in a patch', fontdict={'size': 35}, fontweight='bold')
        # plt.grid(linewidth = 1.5)
        # plt.xticks(fontsize=25)
        # plt.yticks(fontsize=25)
        # plt.show()
        # plt.subplot(122)
        # plt.xlabel('Band Numbers', fontdict={'size': 35}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 35}, fontweight='bold')
        # plt.plot(direction_2[0, :, 0].cpu().detach().numpy(), label='(0,0),0')
        # plt.plot(direction_2[0, :, 1].cpu().detach().numpy(), label='(1,0),5')
        # plt.plot(direction_2[0, :, 2].cpu().detach().numpy(), label='(2,0),10')
        # plt.plot(direction_2[0, :, 3].cpu().detach().numpy(), label='(3,0),15')
        # plt.plot(direction_2[0, :, 4].cpu().detach().numpy(), label='(4,0),20')
        # plt.plot(direction_2[0, :, 5].cpu().detach().numpy(), label='(4,1),21')
        # plt.plot(direction_2[0, :, 6].cpu().detach().numpy(), label='(3,1),16')
        # plt.plot(direction_2[0, :, 7].cpu().detach().numpy(), label='(2,1),11')
        # plt.plot(direction_2[0, :, 8].cpu().detach().numpy(), label='(1,1),6')
        # plt.plot(direction_2[0, :, 9].cpu().detach().numpy(), label='(0,1),1')
        # plt.plot(direction_2[0, :, 10].cpu().detach().numpy(), label='(0,2),2')
        # plt.plot(direction_2[0, :, 11].cpu().detach().numpy(), label='(1,2),7')
        # plt.plot(direction_2[0, :, 12].cpu().detach().numpy(), label='(2,2), center', linewidth=3, linestyle='-.')
        # plt.plot(direction_2[0, :, 13].cpu().detach().numpy(), label='(3,2),17')
        # plt.plot(direction_2[0, :, 14].cpu().detach().numpy(), label='(4,2),22')
        # plt.plot(direction_2[0, :, 15].cpu().detach().numpy(), label='(4,3),23')
        # plt.plot(direction_2[0, :, 16].cpu().detach().numpy(), label='(3,3),18')
        # plt.plot(direction_2[0, :, 17].cpu().detach().numpy(), label='(2,3),13')
        # plt.plot(direction_2[0, :, 18].cpu().detach().numpy(), label='(1,3),8')
        # plt.plot(direction_2[0, :, 19].cpu().detach().numpy(), label='(0,3),3', linewidth=5)
        # plt.plot(direction_2[0, :, 20].cpu().detach().numpy(), label='(0,4),4', linewidth=5)
        # plt.plot(direction_2[0, :, 21].cpu().detach().numpy(), label='(1,4),9', linewidth=5)
        # plt.plot(direction_2[0, :, 22].cpu().detach().numpy(), label='(2,4),14')
        # plt.plot(direction_2[0, :, 23].cpu().detach().numpy(), label='(3,4),19')
        # plt.plot(direction_2[0, :, 24].cpu().detach().numpy(), label='(4,4),24')
        # plt.legend(prop={'size': 15}, fontsize='large', loc=[1, 0])
        # plt.title('Spectral signatures in a patch', fontdict={'size': 35}, fontweight='bold')
        # plt.xticks(fontsize=25)
        # plt.yticks(fontsize=25)
        # plt.show()

        # # plt.subplot(332)
        # plt.imshow(direction_1[0, :, (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-1 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(333)
        # plt.imshow(direction_2[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-2 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(334)
        # plt.imshow(direction_3[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-3 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(335)
        # plt.imshow(direction_4[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-4 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(336)
        # plt.imshow(direction_5[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-5 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(337)
        # plt.imshow(direction_6[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-6 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(338)
        # plt.imshow(direction_7[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()ticks(fontsize=20)
        # plt.title('Direction-7 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(339)
        # plt.imshow(direction_8[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-8 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()

        #换成输入顺序
        x8r = direction_8.permute(2, 0, 1)
        x7r = direction_7.permute(2, 0, 1)
        x6r = direction_6.permute(2, 0, 1)
        x5r = direction_5.permute(2, 0, 1)
        x4r = direction_4.permute(2, 0, 1)
        x3r = direction_3.permute(2, 0, 1)
        x2r = direction_2.permute(2, 0, 1)
        x1r = direction_1.permute(2, 0, 1)
        print('d5.shape', x5r.shape)

        print('x1r',x1r.shape) #(x,b,c) (25,100,204)

        #linear embedding
        x1r = self.linearembedding(x1r)
        x2r = self.linearembedding(x2r)
        x3r = self.linearembedding(x3r)
        x4r = self.linearembedding(x4r)
        x5r = self.linearembedding(x5r)
        x6r = self.linearembedding(x6r)
        x7r = self.linearembedding(x7r)
        x8r = self.linearembedding(x8r) #（25,,100 64)
        #
        # #position_ids
        # # seq_len, batch_size, feature_dim = x1r.shape[0], x1r.shape[1], x1r.shape[2]
        # # print('...', seq_len)
        # # position_ids = torch.arange(0, seq_len).to(device='cuda')
        # # print('...',position_ids.shape)
        # # position_ids = repeat(position_ids, ' n -> b n 1', b=batch_size)
        # # print('...',position_ids.shape)
        #
        # #one_hot
        # # onehot_encoding = F.one_hot(torch.arange(0,seq_len), num_classes=25).to(device='cuda')
        # # print('onehot.shape', onehot_encoding.shape)
        # # onehot_encoding = repeat(onehot_encoding, 'x y -> b x y', b=batch_size)
        # # print('onehot.shape', onehot_encoding.shape) #100, 25, 25
        # # # plt.imshow(onehot_encoding[1,:,:].cpu().detach().numpy())
        #
        # # onehot_encoding_1 = dpe_1
        # # # plt.subplot(241)
        # # # plt.imshow(onehot_encoding_1[1, :, :].cpu().detach().numpy())
        # # onehot_encoding_2 =  dpe_2
        # # # plt.subplot(242)
        # # # plt.imshow(onehot_encoding_2[1, :, :].cpu().detach().numpy())
        # # onehot_encoding_3 = dpe_3
        # # # plt.subplot(243)
        # # # plt.imshow(onehot_encoding_3[1, :, :].cpu().detach().numpy())
        # # onehot_encoding_4 = dpe_4
        # # # plt.subplot(244)
        # # # plt.imshow(onehot_encoding_4[1, :, :].cpu().detach().numpy())
        # # onehot_encoding_5 =  dpe_5
        # # # plt.subplot(245)
        # # # plt.imshow(onehot_encoding_5[1, :, :].cpu().detach().numpy())
        # # onehot_encoding_6 = dpe_6
        # # # plt.subplot(246)
        # # # plt.imshow(onehot_encoding_6[1, :, :].cpu().detach().numpy())
        # # onehot_encoding_7 = dpe_7
        # # # plt.subplot(247)
        # # # plt.imshow(onehot_encoding_7[1, :, :].cpu().detach().numpy())
        # # onehot_encoding_8 = dpe_8
        # # # plt.subplot(248)
        # # # plt.imshow(onehot_encoding_8[1, :, :].cpu().detach().numpy())
        # # # onehot_encoding = self.softmax(onehot_encoding)
        # # # print('onehot + dpe', onehot_encoding[1,:,:])
        # # # plt.show()
        #
        # #positional embedding
        # x1r = rearrange(x1r, 'x b c -> b x c')
        # b, n, c = x1r.shape
        # cls_tokens_x1r = repeat(self.cls_token_1,  '() n c -> b n c', b = b)
        # # cls_tokens_x1r = self.softmax(cls_tokens_x1r)
        # # x1r = x1r * cls_tokens_x1r
        # # x1r = torch.cat((cls_tokens_x1r, x1r), dim=2)
        # # x1r_2_position_ids = torch.cat((position_ids, x1r * 10), dim=2)
        #
        #
        # #sin and cos pos
        # # p_enc_1d_model = PositionalEncoding1D(64)
        # # penc_no_sum = p_enc_1d_model(x1r)
        # # print('penc',penc_no_sum.shape)
        # # x1r_3_sincos = penc_no_sum + x1r
        # # plt.imshow(penc_no_sum[1,:,:].cpu().detach().numpy())
        # # plt.show()
        # # plt.imshow(x1r_3_sincos[1,:,:].cpu().detach().numpy())
        # # plt.show()
        #
        # # x1r_s = x1r + self.pos_embedding_1[:, :(x1r.size(1))]
        # # x1r = torch.cat((onehot_encoding_1, x1r), dim=2)
        # # plt.subplot(241)
        # # plt.imshow(x1r[1, :, :].cpu().detach().numpy())
        # # print('onehot.shape + seq', x1r_s.shape)
        # # plt.imshow(x1r_2_position_ids[1,:,:].cpu().detach().numpy())
        # # plt.colorbar()
        # # plt.show()
        #
        # x2r = rearrange(x2r, 'x b c -> b x c')
        # cls_tokens_x2r = repeat(self.cls_token_2,  '() n c -> b n c', b = b)
        # # cls_tokens_x2r = self.softmax(cls_tokens_x2r)
        # # x2r = x2r * cls_tokens_x2r
        # # x2r = torch.cat((cls_tokens_x2r, x2r), dim=2)
        # # x2r = torch.cat((position_ids, x2r), dim=2)
        # # x2r = torch.cat((onehot_encoding_2, x2r), dim=2)
        # # plt.subplot(242)
        # # plt.imshow(x2r[1, :, :].cpu().detach().numpy())
        #
        # # x2r += self.pos_embedding_2[:, :(x2r.size(1) +1)]
        #
        # x3r = rearrange(x3r, 'x b c -> b x c')
        # cls_tokens_x3r = repeat(self.cls_token_3,  '() n c -> b n c', b = b)
        # # cls_tokens_x3r = self.softmax(cls_tokens_x3r)
        # # x3r = x3r * cls_tokens_x3r
        # # x3r = torch.cat((cls_tokens_x3r, x3r), dim=2)
        # # x3r = torch.cat((position_ids, x3r), dim=2)
        # # x3r = torch.cat((onehot_encoding_3, x3r), dim=2)
        # # plt.subplot(243)
        # # plt.imshow(x3r[1, :, :].cpu().detach().numpy())
        # # x3r += self.pos_embedding_3[:, :(x3r.size(1)+1)]
        #
        # x4r = rearrange(x4r, 'x b c -> b x c')
        # cls_tokens_x4r = repeat(self.cls_token_4,  '() n c -> b n c', b = b)
        # # cls_tokens_x4r = self.softmax(cls_tokens_x4r)
        # # x4r = x4r * cls_tokens_x4r
        # # x4r = torch.cat((cls_tokens_x4r, x4r), dim=2)
        # # x4r = torch.cat((position_ids, x4r), dim=2)
        # # x4r = torch.cat((onehot_encoding_4, x4r), dim=2)
        # # plt.subplot(244)
        # # plt.imshow(x4r[1, :, :].cpu().detach().numpy())
        # # x4r += self.pos_embedding_4[:, :(x4r.size(1)+1)]
        #
        # x5r = rearrange(x5r, 'x b c -> b x c')
        # cls_tokens_x5r = repeat(self.cls_token_5,  '() n c -> b n c', b = b)
        # # cls_tokens_x5r = self.softmax(cls_tokens_x5r)
        # # x5r = x5r * cls_tokens_x5r
        # # x5r = torch.cat((cls_tokens_x5r, x5r), dim=2)
        # # x5r = torch.cat((position_ids, x5r), dim=2)
        # # x5r = torch.cat((onehot_encoding_5, x5r), dim=2)
        # # plt.subplot(245)
        # # plt.imshow(x5r[1, :, :].cpu().detach().numpy())
        # # x5r += self.pos_embedding_5[:, :(x5r.size(1)+1)]
        #
        # x6r = rearrange(x6r, 'x b c -> b x c')
        # cls_tokens_x6r = repeat(self.cls_token_6,  '() n c -> b n c', b = b)
        # # cls_tokens_x6r = self.softmax(cls_tokens_x6r)
        # # x6r = x6r * cls_tokens_x6r
        # # x6r = torch.cat((cls_tokens_x6r, x6r), dim=2)
        # # x6r = torch.cat((position_ids, x6r), dim=2)
        # # x6r = torch.cat((onehot_encoding_6, x6r), dim=2)
        # # plt.subplot(246)
        # # plt.imshow(x6r[1, :, :].cpu().detach().numpy())
        # # x6r += self.pos_embedding_6[:, :(x6r.size(1)+1)]
        #
        # x7r = rearrange(x7r, 'x b c -> b x c')
        # cls_tokens_x7r = repeat(self.cls_token_7,  '() n c -> b n c', b = b)
        # # cls_tokens_x7r = self.softmax(cls_tokens_x7r)
        # # x7r = x7r * cls_tokens_x7r
        # # x7r = torch.cat((cls_tokens_x7r, x7r), dim=2)
        # # x7r = torch.cat((position_ids, x7r), dim=2)
        # # x7r = torch.cat((onehot_encoding_7, x7r), dim=2)
        # # plt.subplot(247)
        # # plt.imshow(x7r[1, :, :].cpu().detach().numpy())
        # # x7r += self.pos_embedding_7[:, :(x7r.size(1)+1)]
        #
        # x8r = rearrange(x8r, 'x b c -> b x c')
        # cls_tokens_x8r = repeat(self.cls_token_8,  '() n c -> b n c', b = b)
        # # cls_tokens_x8r = self.softmax(cls_tokens_x8r)
        # # x8r = x8r * cls_tokens_x8r
        # # x8r = torch.cat((cls_tokens_x8r, x8r), dim=2)
        # # x8r = torch.cat((position_ids, x8r), dim=2)
        # # x8r = torch.cat((onehot_encoding_8, x8r), dim=2)
        # # plt.subplot(248)
        # # plt.imshow(x8r[1, :, :].cpu().detach().numpy())
        # # x8r += self.pos_embedding_8[:, :(x8r.size(1)+1)]
        # # plt.show()
        #
        # x1r = rearrange(x1r, 'b x c -> x b c') #(100, 25, 64+1)-->(25, 100, 64+1)
        # x2r = rearrange(x2r, 'b x c -> x b c')
        # x3r = rearrange(x3r, 'b x c -> x b c')
        # x4r = rearrange(x4r, 'b x c -> x b c')
        # x5r = rearrange(x5r, 'b x c -> x b c')
        # x6r = rearrange(x6r, 'b x c -> x b c')
        # x7r = rearrange(x7r, 'b x c -> x b c')
        # x8r = rearrange(x8r, 'b x c -> x b c')


        '设置transformer的参数给每个direction'
        encoder_layer_1 = nn.TransformerEncoderLayer(d_model=64, nhead=1, dim_feedforward=64,activation='gelu').to(device='cuda')
        transformer_encoder_1 = nn.TransformerEncoder(encoder_layer_1, num_layers=1, norm=None).to(device='cuda')

        encoder_layer_2 = nn.TransformerEncoderLayer(d_model=64, nhead=1, dim_feedforward=64, activation='gelu').to(
            device='cuda')
        transformer_encoder_2 = nn.TransformerEncoder(encoder_layer_2, num_layers=1, norm=None).to(device='cuda')

        encoder_layer_3 = nn.TransformerEncoderLayer(d_model=64, nhead=1, dim_feedforward=64, activation='gelu').to(
            device='cuda')
        transformer_encoder_3 = nn.TransformerEncoder(encoder_layer_3, num_layers=1, norm=None).to(device='cuda')

        encoder_layer_4 = nn.TransformerEncoderLayer(d_model=64, nhead=1, dim_feedforward=64, activation='gelu').to(
            device='cuda')
        transformer_encoder_4 = nn.TransformerEncoder(encoder_layer_4, num_layers=1, norm=None).to(device='cuda')

        encoder_layer_5 = nn.TransformerEncoderLayer(d_model=64, nhead=1, dim_feedforward=64, activation='gelu').to(
            device='cuda')
        transformer_encoder_5 = nn.TransformerEncoder(encoder_layer_5, num_layers=1, norm=None).to(device='cuda')

        encoder_layer_6 = nn.TransformerEncoderLayer(d_model=64, nhead=1, dim_feedforward=64, activation='gelu').to(
            device='cuda')
        transformer_encoder_6 = nn.TransformerEncoder(encoder_layer_6, num_layers=1, norm=None).to(device='cuda')

        encoder_layer_7 = nn.TransformerEncoderLayer(d_model=64, nhead=1, dim_feedforward=64, activation='gelu').to(
            device='cuda')
        transformer_encoder_7 = nn.TransformerEncoder(encoder_layer_7, num_layers=1, norm=None).to(device='cuda')

        encoder_layer_8 = nn.TransformerEncoderLayer(d_model=64, nhead=1, dim_feedforward=64, activation='gelu').to(
            device='cuda')
        transformer_encoder_8 = nn.TransformerEncoder(encoder_layer_8, num_layers=1, norm=None).to(device='cuda')


        #训练transformer
        x1r_output = transformer_encoder_1(x1r)
        # x1r_output = self.layernorm(x1r_output)
        # x1r_output = self.denselayer(x1r_output)

        x2r_output = transformer_encoder_2(x2r)
        # x2r_output = self.layernorm(x2r_output)
        # x2r_output = self.denselayer(x2r_output)

        x3r_output = transformer_encoder_3(x3r)
        # x3r_output = self.layernorm(x3r_output)
        # x3r_output = self.denselayer(x3r_output)

        x4r_output = transformer_encoder_4(x4r)
        # x4r_output = self.layernorm(x4r_output)
        # x4r_output = self.denselayer(x4r_output)

        x5r_output = transformer_encoder_5(x5r)
        # x5r_output = self.layernorm(x5r_output)
        # x5r_output = self.denselayer(x5r_output)

        x6r_output = transformer_encoder_6(x6r)
        # x6r_output = self.layernorm(x6r_output)
        # x6r_output = self.denselayer(x6r_output)

        x7r_output = transformer_encoder_7(x7r)
        # x7r_output = self.layernorm(x7r_output)
        # x7r_output = self.denselayer(x7r_output)

        x8r_output = transformer_encoder_8(x8r)
        # x8r_output = self.layernorm(x8r_output)
        # x8r_output = self.denselayer(x8r_output)
        print('1111', x1r_output.shape) #(x,b,c) SA(25,100,self.embed_dim)
        steps = int(x1r_output.size(0)-1)

        '----show attetntion function------------------------------------------------------'
        def showattention(inputseq):
            allpixel = inputseq[:, 1, :]
            linear1 = nn.Linear(allpixel.size(1),allpixel.size(1)).to( device='cuda')
            # allpixel = linear1(allpixel)

            centralstep = allpixel[int(steps/2),:]
            centralstep = linear1(centralstep)
            # laststep = allpixel[-1, :]
            # laststep = linear1(laststep)

            output = torch.matmul(allpixel, centralstep.transpose(0,-1))

            pairdis = nn.PairwiseDistance()
            cos = nn.CosineSimilarity(dim=-1)

            output_pair = pairdis(allpixel,centralstep) * -1
            # output_pair = cos(allpixel, centralstep)
            # output_pair = torch.matmul(allpixel, centralstep)

            softmax = nn.Softmax()
            output = softmax(output)
            output_pair = softmax(output_pair)
            output = output.unsqueeze(0)
            output_pair = output_pair.unsqueeze(0)
            print('cos',output_pair.shape)
            return output,output_pair
        '------------------------------------------------------------------------------------'
        output1_1 = showattention(x1r_output)[1]
        output1_1_image = reduce(output1_1, 'v (h w) -> h w', h=self.patch_size, reduction='mean')
        vis.heatmap(output1_1_image[:, :].cpu().detach().numpy(),opts={'title': "derection 1"})
        # plt.show()
        # sns.lineplot(data=output1_1.transpose(1,0).cpu().detach().numpy(), markers=True, lw=2)
        # plt.show()
        output2_2 = showattention(x2r_output)[1]
        output2_2_image = reduce(output2_2, 'v (h w) -> h w', h=self.patch_size, reduction='mean')
        vis.heatmap(output2_2_image[:, :].cpu().detach().numpy(), opts={'title': "derection 2"})
        # sns.lineplot(data=output2_2.transpose(1, 0).cpu().detach().numpy(), markers=True, lw=2)
        # plt.show()
        output3_3 = showattention(x3r_output)[1]
        output3_3_image = reduce(output3_3, 'v (h w) -> h w', h=self.patch_size, reduction='mean')
        vis.heatmap(output3_3_image[:, :].cpu().detach().numpy(), opts={'title': "derection 3"})
        # sns.lineplot(data=output3_3.transpose(1, 0).cpu().detach().numpy(), markers=True, lw=2)
        # plt.show()
        output4_4 = showattention(x4r_output)[1]
        output4_4_image = reduce(output4_4, 'v (h w) -> h w', h=self.patch_size, reduction='mean')
        vis.heatmap(output4_4_image[:, :].cpu().detach().numpy(), opts={'title': "derection 4"})
        # sns.lineplot(data=output4_4.transpose(1, 0).cpu().detach().numpy(), markers=True, lw=2)
        # plt.show()
        output5_5 = showattention(x5r_output)[1]
        output5_5_image = reduce(output5_5, 'v (h w) -> h w', h=self.patch_size, reduction='mean')
        vis.heatmap(output5_5_image[:, :].cpu().detach().numpy(), opts={'title': "derection 5"})
        # sns.lineplot(data=output5_5.transpose(1, 0).cpu().detach().numpy(), markers=True, lw=2)
        # plt.show()
        output6_6 = showattention(x6r_output)[1]
        output6_6_image = reduce(output6_6, 'v (h w) -> h w', h=self.patch_size, reduction='mean')
        vis.heatmap(output6_6_image[:, :].cpu().detach().numpy(), opts={'title': "derection 6"})
        # sns.lineplot(data=output6_6.transpose(1, 0).cpu().detach().numpy(), markers=True, lw=2)
        # plt.show()
        output7_7 = showattention(x7r_output)[1]
        output7_7_image = reduce(output7_7, 'v (h w) -> h w', h=self.patch_size, reduction='mean')
        vis.heatmap(output7_7_image[:, :].cpu().detach().numpy(), opts={'title': "derection 7"})
        # sns.lineplot(data=output7_7.transpose(1, 0).cpu().detach().numpy(), markers=True, lw=2)
        # plt.show()
        output8_8 = showattention(x8r_output)[1]
        output8_8_image = reduce(output8_8, 'v (h w) -> h w', h=self.patch_size, reduction='mean')
        vis.heatmap(output8_8_image[:, :].cpu().detach().numpy(), opts={'title': "derection 8"})
        # sns.lineplot(data=output8_8.transpose(1, 0).cpu().detach().numpy(), markers=True, lw=2)
        # plt.show()
        '----------------------------------------------------------------------------'
        outputall = torch.cat([output1_1, output2_2, output3_3, output4_4, output5_5, output6_6, output7_7, output8_8],dim=0)
        sns.lineplot(data=outputall.transpose(1,0).cpu().detach().numpy(), markers=True, lw=2)
        plt.show()
        # all = sns.heatmap(data=outputall.cpu().detach().numpy(), cmap="mako", linewidths=0.05, square=True)
        # all.set_title('all')
        # plt.show()
        '-------------------------------------------------------------------------------------------'
        #提取中间像素信息

        # x1r_output_central = x1r_output[0, :, :] + x1r_output[13,:,:]
        # x1r_output_clstoken = x1r_output[0, :, :]
        x1r_output_centraltoken = x1r_output[int(steps/2), :, :]
        x1r_output_meantoken = reduce(x1r_output, 'x b c -> b c', reduction='mean')
        # print('1111',x1r_output_centraltoken.shape) #(b,c) SA(100, 204)\

        # x2r_output_central = x2r_output[0, :, :] + x2r_output[13,:,:]
        # x2r_output_clstoken = x2r_output[0, :, :]
        x2r_output_centraltoken = x2r_output[int(steps/2),:,:]
        x2r_output_meantoken = reduce(x2r_output, 'x b c -> b c', reduction='mean')

        # x3r_output_central = x3r_output[0, :, :] + x3r_output[13,:,:]
        # x3r_output_clstoken = x3r_output[0, :, :]
        x3r_output_centraltoken = x3r_output[int(steps/2),:,:]
        x3r_output_meantoken = reduce(x3r_output, 'x b c -> b c', reduction='mean')

        # x4r_output_central = x4r_output[0, :, :] + x4r_output[13,:,:]
        # x4r_output_clstoken = x4r_output[0, :, :]
        x4r_output_centraltoken = x4r_output[int(steps/2),:,:]
        x4r_output_meantoken = reduce(x4r_output, 'x b c -> b c', reduction='mean')

        # x5r_output_central = x5r_output[0, :, :] + x5r_output[13,:,:]
        # x5r_output_clstoken = x5r_output[0, :, :]
        x5r_output_centraltoken = x5r_output[int(steps/2),:,:]
        x5r_output_meantoken = reduce(x5r_output, 'x b c -> b c', reduction='mean')

        # x6r_output_central = x6r_output[0, :, :] + x6r_output[13,:,:]
        # x6r_output_clstoken = x6r_output[0, :, :]
        x6r_output_centraltoken = x6r_output[int(steps/2),:,:]
        x6r_output_meantoken = reduce(x6r_output, 'x b c -> b c', reduction='mean')

        # x7r_output_central = x7r_output[0, :, :] + x7r_output[13,:,:]
        # x7r_output_clstoken = x7r_output[0, :, :]
        x7r_output_centraltoken = x7r_output[int(steps/2),:,:]
        x7r_output_meantoken = reduce(x7r_output, 'x b c -> b c', reduction='mean')

        # x8r_output_central = x8r_output[0, :, :] + x8r_output[13,:,:]
        # x8r_output_clstoken = x8r_output[0, :, :]
        x8r_output_centraltoken = x8r_output[int(steps/2),:,:]
        x8r_output_meantoken = reduce(x8r_output, 'x b c -> b c', reduction='mean')

        #扩展维度准备合并
        # x1r_output_centraltoken = rearrange(x1r_output_centraltoken, 'b c -> () b c')
        # x2r_output_centraltoken = rearrange(x2r_output_centraltoken, 'b c -> () b c')
        # x3r_output_centraltoken = rearrange(x3r_output_centraltoken, 'b c -> () b c')
        # x4r_output_centraltoken = rearrange(x4r_output_centraltoken, 'b c -> () b c')
        # x5r_output_centraltoken = rearrange(x5r_output_centraltoken, 'b c -> () b c')
        # x6r_output_centraltoken = rearrange(x6r_output_centraltoken, 'b c -> () b c')
        # x7r_output_centraltoken = rearrange(x7r_output_centraltoken, 'b c -> () b c')
        # x8r_output_centraltoken = rearrange(x8r_output_centraltoken, 'b c -> () b c')
        # print('x1r_output_centraltoken', x1r_output_centraltoken.shape)
        #
        # x1r_output_meantoken = rearrange(x1r_output_meantoken, 'b c -> () b c')
        # x2r_output_meantoken = rearrange(x2r_output_meantoken, 'b c -> () b c')
        # x3r_output_meantoken = rearrange(x3r_output_meantoken, 'b c -> () b c')
        # x4r_output_meantoken = rearrange(x4r_output_meantoken, 'b c -> () b c')
        # x5r_output_meantoken = rearrange(x5r_output_meantoken, 'b c -> () b c')
        # x6r_output_meantoken = rearrange(x6r_output_meantoken, 'b c -> () b c')
        # x7r_output_meantoken = rearrange(x7r_output_meantoken, 'b c -> () b c')
        # x8r_output_meantoken = rearrange(x8r_output_meantoken, 'b c -> () b c')
        # print('x1r_output_meantoken', x1r_output_meantoken.shape)

        '只用一个扫描 or 2 or 4 '
        # preds_onedirection = torch.cat([x1r_output_centraltoken+x1r_output_clstoken,x2r_output_centraltoken+x2r_output_clstoken,x3r_output_centraltoken+x3r_output_clstoken,x4r_output_centraltoken+x4r_output_clstoken
        #                                 ,x5r_output_centraltoken+x5r_output_clstoken,x6r_output_centraltoken+x6r_output_clstoken,x7r_output_centraltoken+x7r_output_clstoken,x8r_output_centraltoken+x8r_output_clstoken], dim=1)
        # preds_onedirection = x1r_output_clstoken+x2r_output_clstoken+x3r_output_clstoken+x4r_output_clstoken
        #                                    +x5r_output_clstoken+x6r_output_clstoken+x7r_output_clstoken+x8r_output_clstoken
        # x1r_output_conca = torch.cat([x1r_output_meantoken,x1r_output_centraltoken],dim=1)
        # x7r_output_conca = torch.cat([x7r_output_meantoken,x7r_output_centraltoken],dim=1)
        # x2r_output_conca = torch.cat([x2r_output_meantoken,x2r_output_centraltoken],dim=1)
        # x8r_output_conca = torch.cat([x8r_output_meantoken,x8r_output_centraltoken],dim=1)
        # x3r_output_conca = torch.cat([x3r_output_meantoken, x3r_output_centraltoken], dim=1)
        # x4r_output_conca = torch.cat([x4r_output_meantoken, x4r_output_centraltoken], dim=1)
        # x5r_output_conca = torch.cat([x5r_output_meantoken, x5r_output_centraltoken], dim=1)
        # x6r_output_conca = torch.cat([x6r_output_meantoken, x6r_output_centraltoken], dim=1)

        x1r_output_conca = repeat(x1r_output_centraltoken, 'b c -> b c ()')
        print('x1r_conca',x1r_output_conca.shape)
        x7r_output_conca = repeat(x7r_output_centraltoken, 'b c -> b c ()')
        x2r_output_conca = repeat(x2r_output_centraltoken, 'b c -> b c ()')
        x8r_output_conca = repeat(x8r_output_centraltoken, 'b c -> b c ()')
        x3r_output_conca = repeat(x3r_output_centraltoken, 'b c -> b c ()')
        x4r_output_conca = repeat(x4r_output_centraltoken, 'b c -> b c ()')
        x5r_output_conca = repeat(x5r_output_centraltoken, 'b c -> b c ()')
        x6r_output_conca = repeat(x6r_output_centraltoken, 'b c -> b c ()')

        preds_onedirection = torch.cat([x1r_output_conca,x7r_output_conca,x2r_output_conca,x8r_output_conca,x3r_output_conca,x5r_output_conca,x4r_output_conca,x6r_output_conca],dim=2)  #（b c x)


        # # x1r = self.gru_2_1(x1r)[0]
        # print('x1r', x1r.shape)  #(x,b,c) (25,100,64)
        # x1r_laststep = x1r[-1]
        # x1r_laststep = self.relu(x1r_laststep)
        # x1r_laststep = torch.unsqueeze(x1r_laststep, dim=0)


        # print('```',preds_onedirection.shape)
        #

        preds_onedirection = preds_onedirection.view(preds_onedirection.size(0), -1)
        preds_onedirection = self.transformer_bn_scheme1(preds_onedirection)
        preds_onedirection = self.relu(preds_onedirection)
        preds_onedirection = self.dropout(preds_onedirection)
        preds_onedirection = self.fc_transformer_scheme1(preds_onedirection)

        "用LSTM"
        # h0_x1r = torch.zeros(1, x1r.size(1), 64).to(device='cuda')
        # c0_x1r = torch.zeros(1, x1r.size(1), 64).to(device="cuda")
        # x1r, (hn_x1r, cn_x1r) = self.lstm_2_1(x1r, (h0_x1r, c0_x1r))
        # # x1r = self.gru_2_1(x1r)[0]
        # print('x1r', x1r.shape)  #(x,b,c) (25,100,64)
        # x1r_laststep = x1r[-1]
        # x1r_laststep = self.relu(x1r_laststep)
        # x1r_laststep = torch.unsqueeze(x1r_laststep, dim=0)
        #
        # h0_x2r = torch.zeros(1, x2r.size(1), 64).to(device='cuda')
        # c0_x2r = torch.zeros(1, x2r.size(1), 64).to(device="cuda")
        # x2r, (hn_x2r, cn_x2r) = self.lstm_2_2(x2r, (h0_x2r, c0_x2r))
        # # x2r = self.gru_2_2(x2r)[0] #把x1r经过RNN的值，作为x2r的输入
        # x2r_laststep = x2r[-1]
        # x2r_laststep = self.relu(x2r_laststep)
        # x2r_laststep = torch.unsqueeze(x2r_laststep, dim=0)
        #
        # h0_x3r = torch.zeros(1, x3r.size(1), 64).to(device='cuda')
        # c0_x3r = torch.zeros(1, x3r.size(1), 64).to(device="cuda")
        # x3r, (hn_x3r, cn_x3r) = self.lstm_2_3(x3r, (h0_x3r, c0_x3r))
        # # x3r = self.gru_2_3(x3r)[0]
        # x3r_laststep = x3r[-1]
        # x3r_laststep = self.relu(x3r_laststep)
        # x3r_laststep = torch.unsqueeze(x3r_laststep, dim=0)
        #
        # h0_x4r = torch.zeros(1, x4r.size(1), 64).to(device='cuda')
        # c0_x4r = torch.zeros(1, x4r.size(1), 64).to(device="cuda")
        # x4r, (hn_x4r, cn_x4r) = self.lstm_2_4(x4r, (h0_x4r, c0_x4r))
        # # x4r = self.gru_2_4(x4r)[0]
        # x4r_laststep = x4r[-1]
        # x4r_laststep = self.relu(x4r_laststep)
        # x4r_laststep = torch.unsqueeze(x4r_laststep, dim=0)
        #
        # h0_x5r = torch.zeros(1, x5r.size(1), 64).to(device='cuda')
        # c0_x5r = torch.zeros(1, x5r.size(1), 64).to(device="cuda")
        # x5r, (hn_x5r, cn_x5r) = self.lstm_2_5(x5r, (h0_x5r, c0_x5r))
        # # x5r = self.gru_2_5(x5r)[0]
        # x5r_laststep = x5r[-1]
        # x5r_laststep = self.relu(x5r_laststep)
        # x5r_laststep = torch.unsqueeze(x5r_laststep, dim=0)
        #
        # h0_x6r = torch.zeros(1, x6r.size(1), 64).to(device='cuda')
        # c0_x6r = torch.zeros(1, x6r.size(1), 64).to(device="cuda")
        # x6r, (hn_x6r, cn_x6r) = self.lstm_2_6(x6r, (h0_x6r, c0_x6r))
        # # x6r = self.gru_2_6(x6r)[0]
        # x6r_laststep = x6r[-1]
        # x6r_laststep = self.relu(x6r_laststep)
        # x6r_laststep = torch.unsqueeze(x6r_laststep, dim=0)
        #
        # h0_x7r = torch.zeros(1, x7r.size(1), 64).to(device='cuda')
        # c0_x7r = torch.zeros(1, x7r.size(1), 64).to(device="cuda")
        # x7r, (hn_x7r, cn_x7r) = self.lstm_2_7(x7r, (h0_x7r, c0_x7r))
        # # x7r = self.gru_2_7(x7r)[0]
        # x7r_laststep = x7r[-1]
        # x7r_laststep = self.relu(x7r_laststep)
        # x7r_laststep = torch.unsqueeze(x7r_laststep, dim=0)
        #
        # h0_x8r = torch.zeros(1, x8r.size(1), 64).to(device='cuda')
        # c0_x8r = torch.zeros(1, x8r.size(1), 64).to(device="cuda")
        # x8r, (hn_x8r, cn_x8r) = self.lstm_2_8(x8r, (h0_x8r, c0_x8r))
        # # x8r = self.gru_2_8(x8r)[0]
        # x8r_laststep = x8r[-1]
        # x8r_laststep = self.relu(x8r_laststep)
        # x8r_laststep = torch.unsqueeze(x8r_laststep, dim=0)
        # print('x8r_last',x8r_laststep.shape)


        "scheme 1"
        # x_strategy_FLC = torch.cat([x1r_output_central,x2r_output_central,x3r_output_central,x4r_output_central,x5r_output_central,x6r_output_central,x7r_output_central,x8r_output_central],dim=0)
        # print('x_strategy_FLC', x_strategy_FLC.shape) # (x, b, c) (8 , batch, 64)
        # x_strategy_FLC = rearrange(x_strategy_FLC, 'x b c -> b x c')
        # x_strategy_FLC += self.pos_embedding_conca_scheme1[:, :(x_strategy_FLC.size(1))]
        # x_strategy_FLC = rearrange(x_strategy_FLC, 'b x c -> x b c')
        # # 设置transformer的参数给8个direction
        # encoder_layer_conca = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=1, dim_feedforward=32, activation='gelu').to(
        #     device='cuda')
        # transformer_encoder_conca = nn.TransformerEncoder(encoder_layer_conca, num_layers=1, norm=None).to(device='cuda')
        #
        # x_strategy_FLC_output = transformer_encoder_conca(x_strategy_FLC)
        # x_strategy_FLC_output = self.denselayer_scheme1(x_strategy_FLC_output)
        # x_strategy_FLC_output = reduce(x_strategy_FLC_output, 'x b c -> 1 b c', reduction='mean')
        # x_strategy_FLC_output = self.relu(x_strategy_FLC_output)
        # x_strategy_FLC_output = x_strategy_FLC_output.view(x_strategy_FLC_output.size(1), -1)
        # x_strategy_FLC_output = self.transformer_bn_scheme1(x_strategy_FLC_output)
        # x_strategy_FLC_output = self.dropout(x_strategy_FLC_output)
        # preds_scheme1 = self.fc_transformer_scheme1(x_strategy_FLC_output)

        "scheme 2"
        # x1_7r = torch.cat([x1r_output_conca, x7r_output_conca], dim=1 )
        # x2_8r = torch.cat([x2r_output_conca, x8r_output_conca], dim=1 )
        # x3_5r = torch.cat([x3r_output_conca, x5r_output_conca], dim=1 )
        # x4_6r = torch.cat([x4r_output_conca, x6r_output_conca], dim=1 )
        # x_strategy_FLC = torch.cat([x1_7r, x2_8r, x3_5r, x4_6r], dim=2)
        # x_strategy_FLC = rearrange(x_strategy_FLC, 'b c x -> b x c')
        # print('...', x_strategy_FLC.shape)
        # cls_tokens_FLC = repeat(self.cls_token_FLC, '() n c -> b n c', b=b)
        # x_strategy_FLC = torch.cat((cls_tokens_FLC, x_strategy_FLC), dim=1)
        # x_strategy_FLC += self.pos_embedding_conca_scheme2[:, :(x_strategy_FLC.size(1) + 1 )]
        # x_strategy_FLC = rearrange(x_strategy_FLC, 'b x c -> x b c')
        # # 设置transformer的参数给4对directions
        # encoder_layer_conca = nn.TransformerEncoderLayer(d_model=self.embed_dim * 4, nhead=1, dim_feedforward=64,
        #                                                  activation='gelu').to(device='cuda')
        # transformer_encoder_conca = nn.TransformerEncoder(encoder_layer_conca, num_layers=1, norm=None).to(device='cuda')
        # x_strategy_FLC_output = transformer_encoder_conca(x_strategy_FLC)
        # x_strategy_FLC_output = self.denselayer_scheme2(x_strategy_FLC_output)
        #
        # x_strategy_FLC_output_clstoken = x_strategy_FLC_output[0,:,:]
        # x_strategy_FLC_output_meantoken = (x_strategy_FLC_output[1,:,:] + x_strategy_FLC_output[2,:,:] + x_strategy_FLC_output[3,:,:] + x_strategy_FLC_output[4,:,:])
        # # x_strategy_FLC_output = reduce(x_strategy_FLC_output, 'x b c -> 1 b c', reduction='mean')
        #
        # x_strategy_FLC_output = self.relu(x_strategy_FLC_output_meantoken)
        # x_strategy_FLC_output = x_strategy_FLC_output.view(x_strategy_FLC_output.size(0), -1)
        # x_strategy_FLC_output = self.transformer_bn_scheme2(x_strategy_FLC_output)
        # x_strategy_FLC_output = self.dropout(x_strategy_FLC_output)
        # preds_scheme2 = self.fc_transformer_scheme2(x_strategy_FLC_output)

        return preds_onedirection


class zhou_slidingLSTM_Trans(nn.Module):

    @staticmethod
    def weight_init(m):
        # All weight matrices in our RNN and bias vectors are initialized with a uniform distribution, and the values of these weight matrices and bias vectors are initialized in the range [−0.1,0.1]
        if isinstance(m, (nn.Linear, nn.GRU, nn.Conv3d, nn.LSTM, nn.TransformerEncoderLayer, nn.TransformerEncoder)):
            init.kaiming_uniform_(m.weight.data)
            init.kaiming_uniform_(m.bias.data)

    def __init__(self, input_channels, n_classes, patch_size=5, embed_dim=64):
        # The proposed network model uses a single recurrent layer that adopts our modified GRUs of size 64 with sigmoid gate activation and PRetanh activation functions for hidden representations
        super(zhou_slidingLSTM_Trans, self).__init__()
        # self.model2 = zhou_dual_LSTM_Trans(input_channels,n_classes,patch_size,embed_dim)
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        'do not use self.n_classes = n_classes'
        self.lstm_trans = nn.LSTM(4, 65 * 4, 1, bidirectional=False)
        self.lstm_2_1 = nn.LSTM(64, 64, 1, bidirectional=False)
        self.lstm_2_2 = nn.LSTM(64, 64, 1, bidirectional=False)
        self.lstm_2_3 = nn.LSTM(64, 64, 1, bidirectional=False)
        self.lstm_2_4 = nn.LSTM(64, 64, 1, bidirectional=False)
        self.lstm_2_5 = nn.LSTM(64, 64, 1, bidirectional=False)
        self.lstm_2_6 = nn.LSTM(64, 64, 1, bidirectional=False)
        self.lstm_2_7 = nn.LSTM(64, 64, 1, bidirectional=False)
        self.lstm_2_8 = nn.LSTM(64, 64, 1, bidirectional=False)
        self.cls_token_1 = nn.Parameter(torch.randn(1, 25, 1))
        self.cls_token_2 = nn.Parameter(torch.randn(1, 25, 1))
        self.cls_token_3 = nn.Parameter(torch.randn(1, 25, 1))
        self.cls_token_4 = nn.Parameter(torch.randn(1, 25, 1))
        self.cls_token_5 = nn.Parameter(torch.randn(1, 25, 1))
        self.cls_token_6 = nn.Parameter(torch.randn(1, 25, 1))
        self.cls_token_7 = nn.Parameter(torch.randn(1, 25, 1))
        self.cls_token_8 = nn.Parameter(torch.randn(1, 25, 1))
        self.lstm1 = nn.LSTMCell(64,64)
        self.pos_embedding_1 = nn.Parameter(torch.randn(1, 25, embed_dim))
        self.pos_embedding_2 = nn.Parameter(torch.randn(1, 25 + 1, embed_dim))
        self.pos_embedding_3 = nn.Parameter(torch.randn(1, 25 + 1, embed_dim))
        self.pos_embedding_4 = nn.Parameter(torch.randn(1, 25 + 1, embed_dim))
        self.pos_embedding_5 = nn.Parameter(torch.randn(1, 25 + 1, embed_dim))
        self.pos_embedding_6 = nn.Parameter(torch.randn(1, 25 + 1, embed_dim))
        self.pos_embedding_7 = nn.Parameter(torch.randn(1, 25 + 1, embed_dim))
        self.pos_embedding_8 = nn.Parameter(torch.randn(1, 25 + 1, embed_dim))
        self.pos_embedding_conca_scheme1 = nn.Parameter(torch.randn(1, 8, embed_dim))
        self.pos_embedding_conca_scheme2 = nn.Parameter(torch.randn(1, 4 + 1, embed_dim * 4))
        self.cls_token_FLC = nn.Parameter(torch.randn(1, 1, embed_dim * 4))
        self.lstm_4 = nn.LSTM(patch_size ** 2, 64, 1)
        self.transformer = nn.Transformer(nhead=16, num_encoder_layers=12)
        self.conv2d_1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=2, stride=1)
        self.dpe = nn.Conv2d(in_channels=input_channels-1 , out_channels=1, kernel_size=1)
        self.dpe_2 = nn.Conv2d(in_channels=input_channels, out_channels=25, kernel_size=1)
        self.depth_conv_1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, padding=1,
                                      groups=input_channels)
        self.point_conv_1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1)
        self.depth_conv_2 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3)
        self.point_conv_2 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3)
        self.conv2d_2 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=2, stride=1)
        self.conv2d_3 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, stride=1)
        self.conv2d_4 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1)
        self.gru_bn = nn.BatchNorm1d(patch_size ** 2 * input_channels)
        self.lstm_bn_1 = nn.BatchNorm1d((64) * 1)
        self.lstm_bn_1_2 = nn.BatchNorm1d((64) * (patch_size) ** 2)
        self.lstm_bn_2 = nn.BatchNorm1d((64) * 8)
        self.lstm_bn_2_2 = nn.BatchNorm1d((64) * 8 * patch_size ** 2)
        self.gru_bn_2 = nn.BatchNorm1d((patch_size ** 2) * (patch_size ** 2))
        self.gru_bn_3 = nn.BatchNorm1d((patch_size ** 2) * (patch_size ** 2) * 2)
        self.gru_bn_4 = nn.BatchNorm1d(8 * 64)
        self.gru_bn_laststep = nn.BatchNorm1d(64)
        self.transformer_bn_scheme1 = nn.BatchNorm1d((embed_dim) * 8)
        self.transformer_bn_scheme2 = nn.BatchNorm1d(embed_dim * 2)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(67600, n_classes)
        self.bn = nn.BatchNorm1d(67600)
        self.lstm_fc_1 = nn.Linear(64 * 1, n_classes)
        self.lstm_fc_1_2 = nn.Linear(64 * (patch_size ** 2), n_classes)
        self.lstm_fc_2 = nn.Linear(64 * 8, n_classes)
        self.lstm_fc_2_2 = nn.Linear(64 * 8 * patch_size ** 2, n_classes)
        self.fc_2 = nn.Linear((patch_size ** 2) * (patch_size ** 2), n_classes)
        self.fc_3 = nn.Linear((patch_size ** 2) * (patch_size ** 2) * 2, n_classes)
        self.fc_4 = nn.Linear(8 * 64, n_classes)
        self.layernorm = nn.LayerNorm(embed_dim)
        self.denselayer = nn.Linear(embed_dim, embed_dim)
        self.denselayer_scheme1 = nn.Linear(embed_dim, embed_dim)
        self.denselayer_scheme2 = nn.Linear(embed_dim * 4, embed_dim * 2)
        self.fc_laststep = nn.Linear(64, n_classes)
        self.fc_transformer_scheme1 = nn.Linear((embed_dim) * 8, n_classes)
        self.fc_transformer_scheme2 = nn.Linear(embed_dim * 2, n_classes)
        self.linearembedding = nn.Linear(in_features=input_channels-1, out_features=embed_dim)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):  # 初始是第1方向

        print('x.shape1', x.shape)

        x = x.squeeze(1)  # b,c,h w
        x_plot = rearrange(x, 'b c h w  -> b (h w) c')

        # for i in range(x_plot.size(1)):
        #     plt.plot(x_plot[1,i,:].cpu().detach().numpy())
        #     plt.xticks(fontsize=20)
        #     plt.yticks(fontsize=20)
        #     plt.grid(visible=True)
        # # plt.show()
        #
        '如果要做TSNE的投影，就激活'
        x_label = x_plot[:,:,0]
        x_label = rearrange(x_label,'b x -> (x b)').cpu().detach().numpy()
        x_label[x_label == 0] = np.nan

        print('type label',x_label)

        x = x[:,1:self.input_channels,:,:]
        x_withoutlabel = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
        x = x_withoutlabel
        # print('x',x)


        '这里做的是soft weight的计算'

        # print('x_label',x_label[1,:])
        # # x = x[:,1:104,:,:]
        # print('x_input',x.shape)
        #
        #
        # x_weight = rearrange(x, 'b c h w  -> b (h w) c')
        #
        # # for i in range(x_weight.size(1)):
        # #     plt.plot(x_weight[1,i,:].cpu().detach().numpy())
        # #     plt.xticks(fontsize=20)
        # #     plt.yticks(fontsize=20)
        # #     plt.grid(visible=True)
        # # plt.show()
        #
        # x_dist = torch.cdist(x_weight, x_weight, p=2)
        # print('distshape', x_dist.shape)
        # mean_x_dist = torch.mean(x_dist)
        # print('meandist', mean_x_dist)
        # print('dist', x_dist.shape)
        # # sns.heatmap(dist[1, :, :].cpu().detach().numpy(),cmap='Blues',linewidths=0.5,annot=True,square=True)
        # # # plt.colorbar()
        # # plt.show()
        #
        # x_weight = torch.exp(-(x_dist ** 2) / mean_x_dist ** 2)
        # print('x_weight',x_weight.shape)
        # # x_weight = repeat(x_weight,'')
        # # weight = self.sigmoid(weight) * 2
        # # g = sns.heatmap(x_weight[1, :, :].cpu().detach().numpy(), cmap='Blues', linewidths=0.5, annot=True, square=True)
        # # g.set_title('original')
        # # # plt.imshow(weight[1, :, :].cpu().detach().numpy(),cmap='blues')
        # # # plt.colorbar()
        # # plt.show()


        # predict = self.model2(x)

        x1 = x  # 用于保存原数据
        x2 = x  # 用于生成dpe
        dpe = self.dpe(x2)
        print(dpe.dtype)
        # print('dpe', dpe)  # (b 1 5 5)

        # 试一试one-hot
        # dpe_onehot = one_hot_extend(dpe)
        #
        # onehotencoder = OneHotEncoder(categories='auto')
        # for i in range(dpe_onehot.size(0)):
        #     i_th = dpe_onehot[i,:]
        #     i_th = i_th.unsqueeze(0)
        #     onehotoutput = onehotencoder.fit_transform(i_th.cpu().detach().numpy())
        #     print(onehotoutput.toarray())
        # i = i + 1
        #
        #

        # multi-dpe (把dpe也换成multiscanning)
        # 生成1和7
        dpe1_0 = dpe[:, :, 0, :]
        dpe1_1 = dpe[:, :, 1, :]
        dpe1_2 = dpe[:, :, 2, :]
        dpe1_3 = dpe[:, :, 3, :]
        dpe1_4 = dpe[:, :, 4, :]
        dpe1_1f = torch.flip(dpe1_1, [2])
        dpe1_3f = torch.flip(dpe1_3, [2])
        # direction_1 = torch.cat([x1_0, x1_1f, x1_2], dim=2)
        dpe_1 = torch.cat([dpe1_0, dpe1_1f, dpe1_2, dpe1_3f, dpe1_4], dim=2)
        print('dpe_1', dpe_1.shape)  # （100, 1 ,25）(b c x)
        dpe_7 = torch.flip(dpe_1, [2])
        dpe_1 = repeat(dpe_1, 'b 1 x -> b x 25')
        dpe_7 = repeat(dpe_7, 'b 1 x -> b x 25')
        print('dpe_1', dpe_1.shape)  # （100, 1 ,25）(b c x)
        # plt.subplot(241)
        # plt.imshow(dpe_1[1, :, :].cpu().detach().numpy())
        # plt.subplot(247)
        # plt.imshow(dpe_7[1, :, :].cpu().detach().numpy())
        # plt.show()

        # 生成第2和8
        dpe2_0 = dpe[:, :, :, 0]
        dpe2_1 = dpe[:, :, :, 1]
        dpe2_2 = dpe[:, :, :, 2]
        dpe2_3 = dpe[:, :, :, 3]
        dpe2_4 = dpe[:, :, :, 4]
        dpe2_1f = torch.flip(dpe2_1, [2])
        dpe2_3f = torch.flip(dpe2_3, [2])
        # direction_2 = torch.cat([x2_0, x2_1f, x2_2], dim=2)
        dpe_2 = torch.cat([dpe2_0, dpe2_1f, dpe2_2, dpe2_3f, dpe2_4], dim=2)
        dpe_8 = torch.flip(dpe_2, [2])
        dpe_2 = repeat(dpe_2, 'b 1 x -> b x 25')
        dpe_8 = repeat(dpe_8, 'b 1 x -> b x 25')
        # plt.subplot(242)
        # plt.imshow(dpe_2[1, :, :].cpu().detach().numpy())
        # plt.subplot(248)
        # plt.imshow(dpe_8[1, :, :].cpu().detach().numpy())

        # 生成3和5
        dpe3_0 = dpe[:, :, 0, :]
        dpe3_1 = dpe[:, :, 1, :]
        dpe3_2 = dpe[:, :, 2, :]
        dpe3_3 = dpe[:, :, 3, :]
        dpe3_4 = dpe[:, :, 4, :]
        dpe3_0f = torch.flip(dpe3_0, [2])
        dpe3_2f = torch.flip(dpe3_2, [2])
        dpe3_4f = torch.flip(dpe3_4, [2])
        # direction_3 = torch.cat([x3_0f, x3_1, x3_2f], dim=2)
        dpe_3 = torch.cat([dpe3_0f, dpe3_1, dpe3_2f, dpe3_3, dpe3_4f], dim=2)
        dpe_5 = torch.flip(dpe_3, [2])
        dpe_3 = repeat(dpe_3, 'b 1 x -> b x 25')
        dpe_5 = repeat(dpe_5, 'b 1 x -> b x 25')
        # plt.subplot(243)
        # plt.imshow(dpe_3[1, :, :].cpu().detach().numpy())
        # plt.subplot(245)
        # plt.imshow(dpe_5[1, :, :].cpu().detach().numpy())

        # 生成4和6
        dpe4_0 = dpe[:, :, :, 0]
        dpe4_1 = dpe[:, :, :, 1]
        dpe4_2 = dpe[:, :, :, 2]
        dpe4_3 = dpe[:, :, :, 3]
        dpe4_4 = dpe[:, :, :, 4]
        dpe4_1f = torch.flip(dpe4_1, [2])
        dpe4_3f = torch.flip(dpe4_3, [2])
        # direction_4 = torch.cat([x4_2, x4_1f, x4_0], dim=2)
        dpe_4 = torch.cat([dpe4_4, dpe4_3f, dpe4_2, dpe4_1f, dpe4_0], dim=2)
        # print('d4', direction_4.shape)
        dpe_6 = torch.flip(dpe_4, [2])
        dpe_4 = repeat(dpe_4, 'b 1 x -> b x 25')
        dpe_6 = repeat(dpe_6, 'b 1 x -> b x 25')
        # plt.subplot(244)
        # plt.imshow(dpe_4[1, :, :].cpu().detach().numpy())
        # plt.subplot(246)
        # plt.imshow(dpe_6[1, :, :].cpu().detach().numpy())
        # plt.show()

        # dpe = reduce(dpe, 'b 1 h w -> b h w', reduction='mean')
        # dpe = rearrange(dpe, 'b h w -> b (h w)')
        # dpe = repeat(dpe, 'b x -> b x 25')
        # print('dpe',dpe.shape)
        # # dpe = self.sigmoid(dpe)
        # plt.subplot(212)
        # plt.imshow(dpe[1,:,:].cpu().detach().numpy())
        # plt.show()
        # print(dpe)
        # xseq = rearrange(x,'b c h w -> b c (h w)')
        # print(xseq.shape)

        # ResNet patch_size = 9 for SA PU
        # x = self.conv2d_1(x)
        # print('1', x.shape)
        # x = self.relu(x)
        # x = self.conv2d_2(x)
        # print('2', x.shape)
        # x_res = self.relu(x)
        # x_res = self.conv2d_3(x_res)
        # print('3', x.shape) #(ptach size = 6)
        # x_res = self.relu(x_res)
        # x_res_res = self.conv2d_4(x_res)
        # x_res_res = self.relu(x_res_res)
        # x = x_res + x_res_res
        # print('4', x.shape)
        # Depthwise separable convolution
        # x1 = self.depth_conv_2(x)
        # x1 = self.point_conv_2(x1)
        # x = self.depth_conv_1(x)
        # x = self.point_conv_1(x)
        # x = self.tanh(x)

        # x = x1 + x

        # x = self.relu(x)
        # x = x * x1

        # x = self.depth_conv_2(x)
        # x = self.point_conv_2(x)

        # 生成第1和7
        x1_0 = x[:, :, 0, :]
        x1_1 = x[:, :, 1, :]
        x1_2 = x[:, :, 2, :]
        x1_3 = x[:, :, 3, :]
        x1_4 = x[:, :, 4, :]
        x1_1f = torch.flip(x1_1, [2])
        x1_3f = torch.flip(x1_3, [2])
        # direction_1 = torch.cat([x1_0, x1_1f, x1_2], dim=2)
        direction_1 = torch.cat([x1_0, x1_1f, x1_2, x1_3f, x1_4], dim=2)

        print('d1', direction_1.shape)  # b c x 100 103 25

        'soft attention mask 速度会变慢'
        # direction_1_cdist = rearrange(direction_1, 'b c x -> b x c')
        # dist = torch.cdist(direction_1_cdist,direction_1_cdist,p=2)
        # # print('distshape',dist.shape)
        # # mean_dist = torch.mean(dist)
        # # print('meandist',mean_dist)
        # # print('dist',dist.shape)
        # sns.heatmap(dist[1, :, :].cpu().detach().numpy(),linewidths=0.5,annot=True,square=True)
        # # # # plt.colorbar()
        # # plt.show()
        # #
        # # weight = torch.exp(-(dist**2) / mean_dist**2)
        # # # weight = self.sigmoid(weight) * 2
        # # # sns.heatmap(weight[1, :, :].cpu().detach().numpy(),cmap='Blues',linewidths=0.5,annot=True,square=True)
        # # # plt.imshow(weight[1, :, :].cpu().detach().numpy(),cmap='blues')
        # # # plt.colorbar()
        # # # plt.show()
        print('d1',direction_1.shape)

        direction_7 = torch.flip(direction_1, [2])

        # 生成第2和8
        x2_0 = x[:, :, :, 0]
        x2_1 = x[:, :, :, 1]
        x2_2 = x[:, :, :, 2]
        x2_3 = x[:, :, :, 3]
        x2_4 = x[:, :, :, 4]
        x2_1f = torch.flip(x2_1, [2])
        x2_3f = torch.flip(x2_3, [2])
        # direction_2 = torch.cat([x2_0, x2_1f, x2_2], dim=2)
        direction_2 = torch.cat([x2_0, x2_1f, x2_2, x2_3f, x2_4], dim=2)
        direction_8 = torch.flip(direction_2, [2])

        # 生成3和5
        x3_0 = x[:, :, 0, :]
        x3_1 = x[:, :, 1, :]
        x3_2 = x[:, :, 2, :]
        x3_3 = x[:, :, 3, :]
        x3_4 = x[:, :, 4, :]
        x3_0f = torch.flip(x3_0, [2])
        x3_2f = torch.flip(x3_2, [2])
        x3_4f = torch.flip(x3_4, [2])
        # direction_3 = torch.cat([x3_0f, x3_1, x3_2f], dim=2)
        direction_3 = torch.cat([x3_0f, x3_1, x3_2f, x3_3, x3_4f], dim=2)
        direction_5 = torch.flip(direction_3, [2])

        # 生成4和6
        x4_0 = x[:, :, :, 0]
        x4_1 = x[:, :, :, 1]
        x4_2 = x[:, :, :, 2]
        x4_3 = x[:, :, :, 3]
        x4_4 = x[:, :, :, 4]
        x4_1f = torch.flip(x4_1, [2])
        x4_3f = torch.flip(x4_3, [2])
        # direction_4 = torch.cat([x4_2, x4_1f, x4_0], dim=2)
        direction_4 = torch.cat([x4_4, x4_3f, x4_2, x4_1f, x4_0], dim=2)
        # print('d4', direction_4.shape)
        direction_6 = torch.flip(direction_4, [2])

        # plt.xlabel('Band Numbers', fontdict={'size': 35}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 35}, fontweight='bold')
        # plt.plot(direction_1[0, :, 0].cpu().detach().numpy(), label='index 0')
        # plt.plot(direction_1[0, :, 1].cpu().detach().numpy(), label='index 1')
        # plt.plot(direction_1[0, :, 2].cpu().detach().numpy(), label='index 2')
        # plt.plot(direction_1[0, :, 3].cpu().detach().numpy(), label='index 3')
        # plt.plot(direction_1[0, :, 4].cpu().detach().numpy(), label='index 4')
        # plt.plot(direction_1[0, :, 5].cpu().detach().numpy(), label='index 9')
        # plt.plot(direction_1[0, :, 6].cpu().detach().numpy(), label='index 8')
        # plt.plot(direction_1[0, :, 7].cpu().detach().numpy(), label='index 7')
        # plt.plot(direction_1[0, :, 8].cpu().detach().numpy(), label='index 6')
        # plt.plot(direction_1[0, :, 9].cpu().detach().numpy(), label='index 5')
        # plt.plot(direction_1[0, :, 10].cpu().detach().numpy(), label='index 10')
        # plt.plot(direction_1[0, :, 11].cpu().detach().numpy(), label='index 11')
        # plt.plot(direction_1[0, :, 12].cpu().detach().numpy(), label='index 12', linewidth=5, linestyle='-.', color = 'red' )
        # plt.plot(direction_1[0, :, 13].cpu().detach().numpy(), label='index 13')
        # plt.plot(direction_1[0, :, 14].cpu().detach().numpy(), label='index 14')
        # plt.plot(direction_1[0, :, 15].cpu().detach().numpy(), label='index 19')
        # plt.plot(direction_1[0, :, 16].cpu().detach().numpy(), label='index 18')
        # plt.plot(direction_1[0, :, 17].cpu().detach().numpy(), label='index 17')
        # plt.plot(direction_1[0, :, 18].cpu().detach().numpy(), label='index 16')
        # plt.plot(direction_1[0, :, 19].cpu().detach().numpy(), label='index 15')
        # plt.plot(direction_1[0, :, 20].cpu().detach().numpy(), label='index 20')
        # plt.plot(direction_1[0, :, 21].cpu().detach().numpy(), label='index 21')
        # plt.plot(direction_1[0, :, 22].cpu().detach().numpy(), label='index 22')
        # plt.plot(direction_1[0, :, 23].cpu().detach().numpy(), label='index 23')
        # plt.plot(direction_1[0, :, 24].cpu().detach().numpy(), label='index 24')
        # plt.legend(prop={'size': 15}, fontsize='large', loc=[1, 0])
        # plt.title('Spectral signatures in a patch', fontdict={'size': 35}, fontweight='bold')
        # plt.grid(linewidth = 1.5)
        # plt.xticks(fontsize=25)
        # plt.yticks(fontsize=25)
        # plt.show()
        # plt.subplot(122)
        # plt.xlabel('Band Numbers', fontdict={'size': 35}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 35}, fontweight='bold')
        # plt.plot(direction_2[0, :, 0].cpu().detach().numpy(), label='(0,0),0')
        # plt.plot(direction_2[0, :, 1].cpu().detach().numpy(), label='(1,0),5')
        # plt.plot(direction_2[0, :, 2].cpu().detach().numpy(), label='(2,0),10')
        # plt.plot(direction_2[0, :, 3].cpu().detach().numpy(), label='(3,0),15')
        # plt.plot(direction_2[0, :, 4].cpu().detach().numpy(), label='(4,0),20')
        # plt.plot(direction_2[0, :, 5].cpu().detach().numpy(), label='(4,1),21')
        # plt.plot(direction_2[0, :, 6].cpu().detach().numpy(), label='(3,1),16')
        # plt.plot(direction_2[0, :, 7].cpu().detach().numpy(), label='(2,1),11')
        # plt.plot(direction_2[0, :, 8].cpu().detach().numpy(), label='(1,1),6')
        # plt.plot(direction_2[0, :, 9].cpu().detach().numpy(), label='(0,1),1')
        # plt.plot(direction_2[0, :, 10].cpu().detach().numpy(), label='(0,2),2')
        # plt.plot(direction_2[0, :, 11].cpu().detach().numpy(), label='(1,2),7')
        # plt.plot(direction_2[0, :, 12].cpu().detach().numpy(), label='(2,2), center', linewidth=3, linestyle='-.')
        # plt.plot(direction_2[0, :, 13].cpu().detach().numpy(), label='(3,2),17')
        # plt.plot(direction_2[0, :, 14].cpu().detach().numpy(), label='(4,2),22')
        # plt.plot(direction_2[0, :, 15].cpu().detach().numpy(), label='(4,3),23')
        # plt.plot(direction_2[0, :, 16].cpu().detach().numpy(), label='(3,3),18')
        # plt.plot(direction_2[0, :, 17].cpu().detach().numpy(), label='(2,3),13')
        # plt.plot(direction_2[0, :, 18].cpu().detach().numpy(), label='(1,3),8')
        # plt.plot(direction_2[0, :, 19].cpu().detach().numpy(), label='(0,3),3', linewidth=5)
        # plt.plot(direction_2[0, :, 20].cpu().detach().numpy(), label='(0,4),4', linewidth=5)
        # plt.plot(direction_2[0, :, 21].cpu().detach().numpy(), label='(1,4),9', linewidth=5)
        # plt.plot(direction_2[0, :, 22].cpu().detach().numpy(), label='(2,4),14')
        # plt.plot(direction_2[0, :, 23].cpu().detach().numpy(), label='(3,4),19')
        # plt.plot(direction_2[0, :, 24].cpu().detach().numpy(), label='(4,4),24')
        # plt.legend(prop={'size': 15}, fontsize='large', loc=[1, 0])
        # plt.title('Spectral signatures in a patch', fontdict={'size': 35}, fontweight='bold')
        # plt.xticks(fontsize=25)
        # plt.yticks(fontsize=25)
        # plt.show()

        # # plt.subplot(332)
        # plt.imshow(direction_1[0, :, (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-1 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(333)
        # plt.imshow(direction_2[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-2 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(334)
        # plt.imshow(direction_3[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-3 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(335)
        # plt.imshow(direction_4[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-4 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(336)
        # plt.imshow(direction_5[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-5 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(337)
        # plt.imshow(direction_6[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-6 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(338)
        # plt.imshow(direction_7[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()ticks(fontsize=20)
        # plt.title('Direction-7 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(339)
        # plt.imshow(direction_8[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-8 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()

        # 换成输入顺序
        x8r = direction_8.permute(2, 0, 1)
        x7r = direction_7.permute(2, 0, 1)
        x6r = direction_6.permute(2, 0, 1)
        x5r = direction_5.permute(2, 0, 1)
        x4r = direction_4.permute(2, 0, 1)
        x3r = direction_3.permute(2, 0, 1)
        x2r = direction_2.permute(2, 0, 1)
        x1r = direction_1.permute(2, 0, 1)
        print('d5.shape', x5r.shape)


        print('x1r', x1r.shape)  # (x,b,c) (25,100,204)


        '2d plot of original data 调配成功'
        # plt.subplot(211)
        # for i in range(x1r.size(0)):
        #     plt.plot(x1r[i,1,:].cpu().detach().numpy())
        #     plt.xticks(fontsize=20)
        #     plt.yticks(fontsize=20)
        #     plt.xlim([0,200])
        #     plt.ylim([0,0.9])
        #     plt.grid(visible=True)
        #
        # plt.subplot(212)
        # plt.imshow(x1r[:,1,:201].cpu().detach().numpy())
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.grid()
        # plt.xlim([0, 200])
        # plt.ylim([0, x1r.size(0)-1])
        # plt.show()

        '3d plot of original data 调配成功'
        # ax = plt.axes(projection='3d')
        # x_data = np.arange(0,103)
        # y_data = np.arange(0,103)
        # for i in range(x1r.size(0)):
        #     z_data = x1r[i, 1, :].cpu().detach().numpy()
        #     ax.plot(x_data, y_data, z_data)
        # plt.show()


        # linear embedding

        '试试sns'
        # print(type(x1r))
        # x1r_numpy = x1r.cpu().numpy()
        # print(type(x1r_numpy),x1r_numpy.shape)
        # x1r_pd = pd.DataFrame(x1r_numpy[:,1,:])
        # print(type(x1r_pd),x1r_pd.head())
        # print('var()',x1r_pd.transpose().var().shape)
        #
        # # sns.relplot(data=x1r_pd.transpose(),kind='line')
        # # plt.show()
        # # plt.xticks(fontsize=20)
        # # plt.yticks(fontsize=20)
        # # matrix = np.tril(x1r_pd.corr())
        # # sns.clustermap(data=x1r_pd)
        # sns.heatmap(x1r_pd.transpose().corr(),square=True,annot=True)
        # plt.show()

        '试试tsne_before'

        # x1r_tsne1 = rearrange(x1r, 'n b c -> (n b) c').cpu().detach().numpy()
        # print('x1r_tsne1',x1r_tsne1.shape)
        # x1r_tsne_out1 = TSNE(n_components=2,verbose=3).fit_transform(x1r_tsne1)
        # print('x1r_tsne_out1',x1r_tsne_out1.shape)
        # plt.subplot(121)
        # plt.subplot(131)
        # plt.scatter(x=x1r_tsne_out[:, 0], y=x1r_tsne_out[:, 1])
        # plt.show()
        '试试kmeans_before'
        # x1r_kmean1 = torch.from_numpy(x1r_tsne_out1)
        # cluster_id_1,cluster_center_1 = kmeans(X=x1r_kmean1,num_clusters=9,distance='euclidean',device=torch.device('cuda'))
        # print('cluster_id_1:',cluster_id_1.shape, 'cluster_center_1:',cluster_center_1.shape)
        # print('id shape',cluster_id_1.shape)
        # im1 = sns.jointplot(x1r_tsne_out1[:,0],x1r_tsne_out1[:,1],hue=x_label,palette='Paired',s=5)
        # print('id_1',cluster_id_1)
        # # sns.relplot(cluster_center_1[:,0],cluster_center_1[:,1],markers='X',
        # #                  alpha=0.6,
        # #                  edgecolors='black',
        # #                  linewidths=2)
        # # im1.set_titles('Original')
        #
        # plt.show()


        x1r = self.linearembedding(x1r)
        x2r = self.linearembedding(x2r)
        x3r = self.linearembedding(x3r)
        x4r = self.linearembedding(x4r)
        x5r = self.linearembedding(x5r)
        x6r = self.linearembedding(x6r)
        x7r = self.linearembedding(x7r)
        x8r = self.linearembedding(x8r)  # （25,,100 64)

        # position_ids
        seq_len, batch_size, feature_dim = x1r.shape[0], x1r.shape[1], x1r.shape[2]
        print('...', seq_len)
        position_ids = torch.arange(0, seq_len).to(device='cuda')
        print('...', position_ids.shape)
        position_ids = repeat(position_ids, ' n -> b n 1', b=batch_size)
        print('...', position_ids.shape)

        # one_hot
        onehot_encoding = F.one_hot(torch.arange(0, seq_len), num_classes=25).to(device='cuda')
        print('onehot.shape', onehot_encoding.shape)
        onehot_encoding = repeat(onehot_encoding, 'x y -> b x y', b=batch_size)
        print('onehot.shape', onehot_encoding.shape)  # 100, 25, 25
        # plt.imshow(onehot_encoding[1,:,:].cpu().detach().numpy())

        onehot_encoding_1 = onehot_encoding + (dpe_1)
        # plt.subplot(241)
        # plt.imshow(onehot_encoding_1[1, :, :].cpu().detach().numpy())
        onehot_encoding_2 = onehot_encoding + (dpe_2)
        # plt.subplot(242)
        # plt.imshow(onehot_encoding_2[1, :, :].cpu().detach().numpy())
        onehot_encoding_3 = onehot_encoding + (dpe_3)
        # plt.subplot(243)
        # plt.imshow(onehot_encoding_3[1, :, :].cpu().detach().numpy())
        onehot_encoding_4 = onehot_encoding + (dpe_4)
        # plt.subplot(244)
        # plt.imshow(onehot_encoding_4[1, :, :].cpu().detach().numpy())
        onehot_encoding_5 = onehot_encoding + (dpe_5)
        # plt.subplot(245)
        # plt.imshow(onehot_encoding_5[1, :, :].cpu().detach().numpy())
        onehot_encoding_6 = onehot_encoding + (dpe_6)
        # plt.subplot(246)
        # plt.imshow(onehot_encoding_6[1, :, :].cpu().detach().numpy())
        onehot_encoding_7 = onehot_encoding + (dpe_7)
        # plt.subplot(247)
        # plt.imshow(onehot_encoding_7[1, :, :].cpu().detach().numpy())
        onehot_encoding_8 = onehot_encoding + (dpe_8)
        # plt.subplot(248)
        # plt.imshow(onehot_encoding_8[1, :, :].cpu().detach().numpy())
        # onehot_encoding = self.softmax(onehot_encoding)
        # print('onehot + dpe', onehot_encoding[1,:,:])
        # plt.show()

        # positional embedding
        x1r = rearrange(x1r, 'x b c -> b x c')
        b, n, c = x1r.shape
        cls_tokens_x1r = repeat(self.cls_token_1, '() n c -> b n c', b=b)
        # cls_tokens_x1r = self.softmax(cls_tokens_x1r)
        # x1r = x1r * cls_tokens_x1r
        # x1r = torch.cat((cls_tokens_x1r, x1r), dim=2)
        # x1r_2_position_ids = torch.cat((position_ids, x1r * 10), dim=2)
        # x1r = torch.cat((onehot_encoding_1, x1r), dim=2)

        # sin and cos pos
        # p_enc_1d_model = PositionalEncoding1D(64)
        # penc_no_sum = p_enc_1d_model(x1r)
        # print('penc', penc_no_sum.shape)
        # x1r_3_sincos = penc_no_sum + x1r
        # plt.imshow(penc_no_sum[1,:,:].cpu().detach().numpy())
        # plt.show()
        # plt.imshow(x1r_3_sincos[1,:,:].cpu().detach().numpy())
        # plt.show()

        # x1r_s = x1r + self.pos_embedding_1[:, :(x1r.size(1))]

        # plt.subplot(241)
        # plt.imshow(x1r[1, :, :].cpu().detach().numpy())
        # print('onehot.shape + seq', x1r_s.shape)
        # plt.imshow(x1r_2_position_ids[1,:,:].cpu().detach().numpy())
        # plt.colorbar()
        # plt.show()

        x2r = rearrange(x2r, 'x b c -> b x c')
        cls_tokens_x2r = repeat(self.cls_token_2, '() n c -> b n c', b=b)
        # cls_tokens_x2r = self.softmax(cls_tokens_x2r)
        # x2r = x2r * cls_tokens_x2r
        # x2r = torch.cat((cls_tokens_x2r, x2r), dim=2)
        # x2r = torch.cat((position_ids, x2r), dim=2)
        # x2r = torch.cat((onehot_encoding_2, x2r), dim=2)
        # plt.subplot(242)
        # plt.imshow(x2r[1, :, :].cpu().detach().numpy())

        # x2r += self.pos_embedding_2[:, :(x2r.size(1) +1)]

        x3r = rearrange(x3r, 'x b c -> b x c')
        cls_tokens_x3r = repeat(self.cls_token_3, '() n c -> b n c', b=b)
        # cls_tokens_x3r = self.softmax(cls_tokens_x3r)
        # x3r = x3r * cls_tokens_x3r
        # x3r = torch.cat((cls_tokens_x3r, x3r), dim=2)
        # x3r = torch.cat((position_ids, x3r), dim=2)
        # x3r = torch.cat((onehot_encoding_3, x3r), dim=2)
        # plt.subplot(243)
        # plt.imshow(x3r[1, :, :].cpu().detach().numpy())
        # x3r += self.pos_embedding_3[:, :(x3r.size(1)+1)]

        x4r = rearrange(x4r, 'x b c -> b x c')
        cls_tokens_x4r = repeat(self.cls_token_4, '() n c -> b n c', b=b)
        # cls_tokens_x4r = self.softmax(cls_tokens_x4r)
        # x4r = x4r * cls_tokens_x4r
        # x4r = torch.cat((cls_tokens_x4r, x4r), dim=2)
        # x4r = torch.cat((position_ids, x4r), dim=2)
        # x4r = torch.cat((onehot_encoding_4, x4r), dim=2)
        # plt.subplot(244)
        # plt.imshow(x4r[1, :, :].cpu().detach().numpy())
        # x4r += self.pos_embedding_4[:, :(x4r.size(1)+1)]

        x5r = rearrange(x5r, 'x b c -> b x c')
        cls_tokens_x5r = repeat(self.cls_token_5, '() n c -> b n c', b=b)
        # cls_tokens_x5r = self.softmax(cls_tokens_x5r)
        # x5r = x5r * cls_tokens_x5r
        # x5r = torch.cat((cls_tokens_x5r, x5r), dim=2)
        # x5r = torch.cat((position_ids, x5r), dim=2)
        # x5r = torch.cat((onehot_encoding_5, x5r), dim=2)
        # plt.subplot(245)
        # plt.imshow(x5r[1, :, :].cpu().detach().numpy())
        # x5r += self.pos_embedding_5[:, :(x5r.size(1)+1)]

        x6r = rearrange(x6r, 'x b c -> b x c')
        cls_tokens_x6r = repeat(self.cls_token_6, '() n c -> b n c', b=b)
        # cls_tokens_x6r = self.softmax(cls_tokens_x6r)
        # x6r = x6r * cls_tokens_x6r
        # x6r = torch.cat((cls_tokens_x6r, x6r), dim=2)
        # x6r = torch.cat((position_ids, x6r), dim=2)
        # x6r = torch.cat((onehot_encoding_6, x6r), dim=2)
        # plt.subplot(246)
        # plt.imshow(x6r[1, :, :].cpu().detach().numpy())
        # x6r += self.pos_embedding_6[:, :(x6r.size(1)+1)]

        x7r = rearrange(x7r, 'x b c -> b x c')
        cls_tokens_x7r = repeat(self.cls_token_7, '() n c -> b n c', b=b)
        # cls_tokens_x7r = self.softmax(cls_tokens_x7r)
        # x7r = x7r * cls_tokens_x7r
        # x7r = torch.cat((cls_tokens_x7r, x7r), dim=2)
        # x7r = torch.cat((position_ids, x7r), dim=2)
        # x7r = torch.cat((onehot_encoding_7, x7r), dim=2)
        # plt.subplot(247)
        # plt.imshow(x7r[1, :, :].cpu().detach().numpy())
        # x7r += self.pos_embedding_7[:, :(x7r.size(1)+1)]

        x8r = rearrange(x8r, 'x b c -> b x c')
        cls_tokens_x8r = repeat(self.cls_token_8, '() n c -> b n c', b=b)
        # cls_tokens_x8r = self.softmax(cls_tokens_x8r)
        # x8r = x8r * cls_tokens_x8r
        # x8r = torch.cat((cls_tokens_x8r, x8r), dim=2)
        # x8r = torch.cat((position_ids, x8r), dim=2)
        # x8r = torch.cat((onehot_encoding_8, x8r), dim=2)
        # plt.subplot(248)
        # plt.imshow(x8r[1, :, :].cpu().detach().numpy())
        # x8r += self.pos_embedding_8[:, :(x8r.size(1)+1)]
        # plt.show()

        x1r = rearrange(x1r, 'b x c -> x b c')  # (100, 25, 64+25)-->(25, 100, 64+25)
        x2r = rearrange(x2r, 'b x c -> x b c')
        x3r = rearrange(x3r, 'b x c -> x b c')
        x4r = rearrange(x4r, 'b x c -> x b c')
        x5r = rearrange(x5r, 'b x c -> x b c')
        x6r = rearrange(x6r, 'b x c -> x b c')
        x7r = rearrange(x7r, 'b x c -> x b c')
        x8r = rearrange(x8r, 'b x c -> x b c')

        #用LSTM做positional encoding and projection

        # hx = torch.zeros(x1r.size(1), 64)
        # cx = torch.zeros(x1r.size(1), 64)
        # output1 = []
        # output1hx = []
        # output1cx = []
        # for i in range(x1r.size(0)):
        #     hx, cx = self.lstm1(x1r[i,:,:], (hx,cx))
        #     output1.append(hx)
        #     output1hx.append(hx)
        #     output1cx.append(cx)
        # output1 = torch.stack(output1,dim=0)
        # output1hx = torch.stack(output1hx,dim=0)
        # output1cx = torch.stack(output1cx,dim=0)
        # print('hn_size:', output1hx.shape, 'cn_size:', output1cx.shape)
        #
        # plt.subplot(131)
        # sns.heatmap(x1r[:,0,:].cpu().detach().numpy())
        # plt.subplot(132)
        # sns.heatmap(output1hx[:,0,:].cpu().detach().numpy())
        # plt.subplot(133)
        # sns.heatmap(output1cx[:, 0, :].cpu().detach().numpy())
        # plt.show()


        h0_x1r = Variable(torch.zeros(1, x1r.size(1), 64)).to(device='cuda') #可以试试torch.zeros或者torch.rand
        c0_x1r = Variable(torch.zeros(1, x1r.size(1), 64)).to(device="cuda") #要不要加requires_grad=True ?
        x1r, (hn_x1r, cn_x1r) = self.lstm_2_1(x1r, (h0_x1r, c0_x1r))

        var1 = torch.var(x1r)
        print('var1:', var1)
        # print('hn_size:', hn_x1r.shape, 'cn_size:', cn_x1r.shape)

        # plt.subplot(131)
        # sns.heatmap(x1r[:,0,:].cpu().detach().numpy(),center=0)
        # plt.subplot(132)
        # sns.heatmap(hn_x1r[:,0,:].cpu().detach().numpy(),center=0)
        # plt.subplot(133)
        # sns.heatmap(cn_x1r[:, 0, :].cpu().detach().numpy(),center=0,)
        # plt.show()
        # x1r = self.gru_2_1(x1r)[0]
        print('x1r', x1r.shape)  #(x,b,c) (25,100,64)
        x1r_laststep = x1r[-1]
        x1r_laststep = self.relu(x1r_laststep)
        x1r_laststep = torch.unsqueeze(x1r_laststep, dim=0)

        't-sne 2'
        x1r_tsne2 = rearrange(x1r, 'n b c -> (n b) c').cpu().detach().numpy()
        # print('x1r_tsne', x1r_tsne.shape)
        x1r_tsne_out2 = TSNE(n_components=2,verbose=3).fit_transform(x1r_tsne2)
        # print('x1r_tsne_out', x1r_tsne_out.shape)
        # plt.subplot(122)
        # plt.subplot(132)
        # plt.scatter(x=x1r_tsne_out2[:, 0], y=x1r_tsne_out2[:, 1])
        im1 = sns.jointplot(x1r_tsne_out2[:, 0], x1r_tsne_out2[:, 1], hue=x_label, palette='Paired', s=5)
        plt.show()
        'kmeans_mid'
        # x1r_kmean2 = torch.from_numpy(x1r_tsne_out2)
        # cluster_id_2,cluster_center_2 = kmeans(X=x1r_kmean2,num_clusters=9,distance='euclidean',device=torch.device('cuda'))
        # print('cluster_id_2:',cluster_id_2.shape, 'cluster_center_2:',cluster_center_2.shape)
        # im2 = sns.jointplot(x1r_tsne_out2[:,0],x1r_tsne_out2[:,1],hue=x_label,palette='Paired',s=5)
        # # plt.scatter(cluster_center_2[:,0],cluster_center_2[:,1],c='white',
        # #                  alpha=0.6,
        # #                  edgecolors='black',
        # #                  linewidths=2)
        # # im2.set_titles('After LSTM')
        # plt.show()

        # direction_2_cdist = rearrange(x1r, 'x b c -> b x c')
        # dist2 = torch.cdist(direction_2_cdist, direction_2_cdist, p=2)
        # print('distshape',dist.shape)
        # mean_dist = torch.mean(dist)
        # print('meandist',mean_dist)
        # print('dist',dist.shape)
        # sns.heatmap(dist2[1, :, :].cpu().detach().numpy(), linewidths=0.5, annot=True, square=True)
        # # # plt.colorbar()
        # plt.show()
        #
        '3d plot of features'
        # ax = plt.axes(projection='3d')
        # x_data = np.arange(0,64)
        # y_data = np.arange(0,64)
        # for i in range(x1r.size(0)):
        #     z_data = x1r[i, 1, :].cpu().detach().numpy()
        #     ax.plot(x_data, y_data, z_data)
        # plt.show()

        # direction_1_cdist = rearrange(x1r, 'x b c -> b x c')
        # dist = torch.cdist(direction_1_cdist,direction_1_cdist,p=2)
        # print('dist',dist.shape)
        # sns.heatmap(dist[0, :, :].cpu().detach().numpy(),linewidths=0.5,annot=True,annot_kws={'fontweight':'bold'},square=True)
        # plt.title('dist')
        # plt.show()
        # weight = -dist
        # weight = weight + 1
        # sns.heatmap(weight[1, :, :].cpu().detach().numpy(),cmap='Blues',linewidths=0.5,annot=True,annot_kws={'fontweight':'bold'},square=True)
        # plt.title('weight')
        # plt.show()
        # print('d1',direction_1.shape)

        # x1rplot = rearrange(x1r, 'l b c -> c b l')
        # plt.subplot(241)
        # plt.plot(x1rplot[:,0,:].cpu().detach().numpy())


        h0_x2r = Variable(torch.zeros(1, x2r.size(1), 64)).to(device='cuda')
        c0_x2r = Variable(torch.zeros(1, x2r.size(1), 64)).to(device="cuda")
        x2r, (hn_x2r, cn_x2r) = self.lstm_2_2(x2r, (h0_x2r, c0_x2r))
        # x2r = self.gru_2_2(x2r)[0] #把x1r经过RNN的值，作为x2r的输入
        x2r_laststep = x2r[-1]
        x2r_laststep = self.relu(x2r_laststep)
        x2r_laststep = torch.unsqueeze(x2r_laststep, dim=0)

        # x2rplot = rearrange(x2r, 'l b c -> c b l')
        # plt.subplot(242)
        # plt.plot(x2rplot[:, 0, :].cpu().detach().numpy())


        h0_x3r = Variable(torch.zeros(1, x3r.size(1), 64)).to(device='cuda')
        c0_x3r = Variable(torch.zeros(1, x3r.size(1), 64)).to(device="cuda")
        x3r, (hn_x3r, cn_x3r) = self.lstm_2_3(x3r, (h0_x3r, c0_x3r))
        # x3r = self.gru_2_3(x3r)[0]
        x3r_laststep = x3r[-1]
        x3r_laststep = self.relu(x3r_laststep)
        x3r_laststep = torch.unsqueeze(x3r_laststep, dim=0)

        # x3rplot = rearrange(x3r, 'l b c -> c b l')
        # plt.subplot(243)
        # plt.plot(x3rplot[:, 0, :].cpu().detach().numpy())

        h0_x4r = Variable(torch.zeros(1, x4r.size(1), 64)).to(device='cuda')
        c0_x4r = Variable(torch.zeros(1, x4r.size(1), 64)).to(device="cuda")
        x4r, (hn_x4r, cn_x4r) = self.lstm_2_4(x4r, (h0_x4r, c0_x4r))
        # x4r = self.gru_2_4(x4r)[0]
        x4r_laststep = x4r[-1]
        x4r_laststep = self.relu(x4r_laststep)
        x4r_laststep = torch.unsqueeze(x4r_laststep, dim=0)

        # x4rplot = rearrange(x4r, 'l b c -> c b l')
        # plt.subplot(244)
        # plt.plot(x4rplot[:, 0, :].cpu().detach().numpy())

        h0_x5r = Variable(torch.zeros(1, x5r.size(1), 64)).to(device='cuda')
        c0_x5r = Variable(torch.zeros(1, x5r.size(1), 64)).to(device="cuda")
        x5r, (hn_x5r, cn_x5r) = self.lstm_2_5(x5r, (h0_x5r, c0_x5r))
        # x5r = self.gru_2_5(x5r)[0]
        x5r_laststep = x5r[-1]
        x5r_laststep = self.relu(x5r_laststep)
        x5r_laststep = torch.unsqueeze(x5r_laststep, dim=0)

        # x5rplot = rearrange(x5r, 'l b c -> c b l')
        # plt.subplot(245)
        # plt.plot(x5rplot[:, 0, :].cpu().detach().numpy())

        h0_x6r = Variable(torch.zeros(1, x6r.size(1), 64)).to(device='cuda')
        c0_x6r = Variable(torch.zeros(1, x6r.size(1), 64)).to(device="cuda")
        x6r, (hn_x6r, cn_x6r) = self.lstm_2_6(x6r, (h0_x6r, c0_x6r))
        # x6r = self.gru_2_6(x6r)[0]
        x6r_laststep = x6r[-1]
        x6r_laststep = self.relu(x6r_laststep)
        x6r_laststep = torch.unsqueeze(x6r_laststep, dim=0)

        # x6rplot = rearrange(x6r, 'l b c -> c b l')
        # plt.subplot(246)
        # plt.plot(x6rplot[:, 0, :].cpu().detach().numpy())

        h0_x7r = Variable(torch.zeros(1, x7r.size(1), 64)).to(device='cuda')
        c0_x7r = Variable(torch.zeros(1, x7r.size(1), 64)).to(device="cuda")
        x7r, (hn_x7r, cn_x7r) = self.lstm_2_7(x7r, (h0_x7r, c0_x7r))
        # x7r = self.gru_2_7(x7r)[0]
        x7r_laststep = x7r[-1]
        x7r_laststep = self.relu(x7r_laststep)
        x7r_laststep = torch.unsqueeze(x7r_laststep, dim=0)

        # x7rplot = rearrange(x7r, 'l b c -> c b l')
        # plt.subplot(247)
        # plt.plot(x7rplot[:, 0, :].cpu().detach().numpy())

        h0_x8r = Variable(torch.zeros(1, x8r.size(1), 64)).to(device='cuda')
        c0_x8r = Variable(torch.zeros(1, x8r.size(1), 64)).to(device="cuda")
        x8r, (hn_x8r, cn_x8r) = self.lstm_2_8(x8r, (h0_x8r, c0_x8r))
        # x8r = self.gru_2_8(x8r)[0]
        x8r_laststep = x8r[-1]
        x8r_laststep = self.relu(x8r_laststep)
        x8r_laststep = torch.unsqueeze(x8r_laststep, dim=0)
        print('x8r_last',x8r_laststep.shape)

        # x8rplot = rearrange(x8r, 'l b c -> c b l')
        # plt.subplot(248)
        # plt.plot(x8rplot[:, 0, :].cpu().detach().numpy())
        # plt.show()



        # 设置transformer的参数给每个direction
        encoder_layer_1 = nn.TransformerEncoderLayer(d_model=64, nhead=1, dim_feedforward=64, activation='gelu').to(
            device='cuda')
        transformer_encoder_1 = nn.TransformerEncoder(encoder_layer_1, num_layers=1, norm=None).to(device='cuda')

        encoder_layer_2 = nn.TransformerEncoderLayer(d_model=64, nhead=1, dim_feedforward=64, activation='gelu').to(
            device='cuda')
        transformer_encoder_2 = nn.TransformerEncoder(encoder_layer_2, num_layers=1, norm=None).to(device='cuda')

        encoder_layer_3 = nn.TransformerEncoderLayer(d_model=64, nhead=1, dim_feedforward=64, activation='gelu').to(
            device='cuda')
        transformer_encoder_3 = nn.TransformerEncoder(encoder_layer_3, num_layers=1, norm=None).to(device='cuda')

        encoder_layer_4 = nn.TransformerEncoderLayer(d_model=64, nhead=1, dim_feedforward=64, activation='gelu').to(
            device='cuda')
        transformer_encoder_4 = nn.TransformerEncoder(encoder_layer_4, num_layers=1, norm=None).to(device='cuda')

        encoder_layer_5 = nn.TransformerEncoderLayer(d_model=64, nhead=1, dim_feedforward=64, activation='gelu').to(
            device='cuda')
        transformer_encoder_5 = nn.TransformerEncoder(encoder_layer_5, num_layers=1, norm=None).to(device='cuda')

        encoder_layer_6 = nn.TransformerEncoderLayer(d_model=64, nhead=1, dim_feedforward=64, activation='gelu').to(
            device='cuda')
        transformer_encoder_6 = nn.TransformerEncoder(encoder_layer_6, num_layers=1, norm=None).to(device='cuda')

        encoder_layer_7 = nn.TransformerEncoderLayer(d_model=64, nhead=1, dim_feedforward=64, activation='gelu').to(
            device='cuda')
        transformer_encoder_7 = nn.TransformerEncoder(encoder_layer_7, num_layers=1, norm=None).to(device='cuda')

        encoder_layer_8 = nn.TransformerEncoderLayer(d_model=64, nhead=1, dim_feedforward=64, activation='gelu').to(
            device='cuda')
        transformer_encoder_8 = nn.TransformerEncoder(encoder_layer_8, num_layers=1, norm=None).to(device='cuda')

        # 训练transformer
        x1r_output = transformer_encoder_1(x1r)

        var2 = torch.var(x1r_output)
        print('var2:', var2)

        'tsne after'
        x1r_tsne3 = rearrange(x1r_output, 'n b c -> (n b) c').cpu().detach().numpy()
        # print('x1r_tsne', x1r_tsne.shape)
        x1r_tsne_out3 = TSNE(n_components=2,verbose=3).fit_transform(x1r_tsne3)
        # print('x1r_tsne_out', x1r_tsne_out.shape)
        # plt.subplot(122)
        # plt.subplot(133)
        # plt.scatter(x=x1r_tsne_out2[:, 0], y=x1r_tsne_out2[:, 1])
        im1 = sns.jointplot(x1r_tsne_out3[:, 0], x1r_tsne_out3[:, 1], hue=x_label, palette='Paired', s=5)
        plt.show()
        'kmeans_after'
        # x1r_kmean3 = torch.from_numpy(x1r_tsne_out3)
        # cluster_id_3,cluster_center_3 = kmeans(X=x1r_kmean3,num_clusters=9,distance='euclidean',device=torch.device('cuda'))
        # print('cluster_id_3:',cluster_id_3.shape, 'cluster_center_3:',cluster_center_3.shape)
        # im3 = sns.jointplot(x1r_tsne_out3[:,0],x1r_tsne_out3[:,1],hue=x_label,palette='Paired',s=5)
        # # plt.scatter(cluster_center_3[:,0],cluster_center_3[:,1],c='white',
        # #                  alpha=0.6,
        # #                  edgecolors='black',
        # #                  linewidths=2)
        # # im3.set_titles('After Transformer')
        # plt.show()

        '看看用完trans之后pixel之间的pair distance'
        # direction_3_cdist = rearrange(x1r_output, 'x b c -> b x c')
        # dist3 = torch.cdist(direction_3_cdist, direction_3_cdist, p=2)
        # # print('distshape',dist.shape)
        # # mean_dist = torch.mean(dist)
        # # print('meandist',mean_dist)
        # # print('dist',dist.shape)
        # sns.heatmap(dist3[1, :, :].cpu().detach().numpy(), linewidths=0.5, annot=True, square=True)
        # # # # plt.colorbar()
        # # plt.show()
        #
        # # x1rplot = rearrange(x1r_output, 'l b c -> c b l')
        # # plt.subplot(241)
        # # plt.plot(x1rplot[:, 0, :].cpu().detach().numpy())
        # # x1r_output = self.layernorm(x1r_output)
        # x1r_output = self.denselayer(x1r_output)

        # direction_2_cdist = rearrange(x1r_output, 'x b c -> b x c')
        # dist2 = torch.cdist(direction_2_cdist,direction_2_cdist,p=2)
        # print('dist',dist2.shape)
        # sns.heatmap(dist2[0, :, :].cpu().detach().numpy(),linewidths=0.5,annot=True,annot_kws={'fontweight':'bold'},square=True)
        # plt.title('dist after trans')
        # plt.show()

        x2r_output = transformer_encoder_2(x2r)

        # x2rplot = rearrange(x2r_output, 'l b c -> c b l')
        # plt.subplot(242)
        # plt.plot(x2rplot[:, 0, :].cpu().detach().numpy())
        # x2r_output = self.layernorm(x2r_output)
        # x2r_output = self.denselayer(x2r_output)

        x3r_output = transformer_encoder_3(x3r)

        # x3rplot = rearrange(x3r_output, 'l b c -> c b l')
        # plt.subplot(243)
        # plt.plot(x3rplot[:, 0, :].cpu().detach().numpy())
        # x3r_output = self.layernorm(x3r_output)
        # x3r_output = self.denselayer(x3r_output)

        x4r_output = transformer_encoder_4(x4r)

        # x4rplot = rearrange(x4r_output, 'l b c -> c b l')
        # plt.subplot(244)
        # plt.plot(x4rplot[:, 0, :].cpu().detach().numpy())
        # x4r_output = self.layernorm(x4r_output)
        # x4r_output = self.denselayer(x4r_output)

        x5r_output = transformer_encoder_5(x5r)

        # x5rplot = rearrange(x5r_output, 'l b c -> c b l')
        # plt.subplot(245)
        # plt.plot(x5rplot[:, 0, :].cpu().detach().numpy())
        # x5r_output = self.layernorm(x5r_output)
        # x5r_output = self.denselayer(x5r_output)

        x6r_output = transformer_encoder_6(x6r)

        # x6rplot = rearrange(x6r_output, 'l b c -> c b l')
        # plt.subplot(246)
        # plt.plot(x6rplot[:, 0, :].cpu().detach().numpy())
        # x6r_output = self.layernorm(x6r_output)
        # x6r_output = self.denselayer(x6r_output)

        x7r_output = transformer_encoder_7(x7r)

        # x7rplot = rearrange(x7r_output, 'l b c -> c b l')
        # plt.subplot(247)
        # plt.plot(x7rplot[:, 0, :].cpu().detach().numpy())
        # x7r_output = self.layernorm(x7r_output)
        # x7r_output = self.denselayer(x7r_output)

        x8r_output = transformer_encoder_8(x8r)

        # x8rplot = rearrange(x8r_output, 'l b c -> c b l')
        # plt.subplot(248)
        # plt.plot(x8rplot[:, 0, :].cpu().detach().numpy())
        # x8r_output = self.layernorm(x8r_output)
        # x8r_output = self.denselayer(x8r_output)
        print('1111', x1r_output.shape)  # (x,b,c) SA(25,100,self.embed_dim)
        # plt.show()
        '试试transformer decoder'
        decoder_layer_1 = nn.TransformerDecoderLayer(d_model=64, nhead=1, dim_feedforward=64, activation='gelu').to(
            device='cuda')
        transformer_decoder_1 = nn.TransformerDecoder(decoder_layer_1, num_layers=1, norm=None).to(device='cuda')
        memory_1 = x1r_output
        target_1 = x1r_output
        x1r_decoder_out = transformer_decoder_1(target_1,memory_1)
        # plt.subplot(121)
        # plt.imshow(x1r_output[:,1,:].cpu().detach().numpy())
        # plt.subplot(122)
        # plt.imshow(x1r_decoder_out[:,1,:].cpu().detach().numpy())
        # plt.show()
        decoder_layer_2 = nn.TransformerDecoderLayer(d_model=64, nhead=1, dim_feedforward=64, activation='gelu').to(
            device='cuda')
        transformer_decoder_2 = nn.TransformerDecoder(decoder_layer_2, num_layers=1, norm=None).to(device='cuda')
        memory_2 = x2r_output
        target_2 = x2r_output
        x2r_decoder_out = transformer_decoder_2(target_2, memory_2)

        decoder_layer_3 = nn.TransformerDecoderLayer(d_model=64, nhead=1, dim_feedforward=64, activation='gelu').to(
            device='cuda')
        transformer_decoder_3 = nn.TransformerDecoder(decoder_layer_3, num_layers=1, norm=None).to(device='cuda')
        memory_3 = x3r_output
        target_3 = x3r_output
        x3r_decoder_out = transformer_decoder_3(target_3, memory_3)

        decoder_layer_4 = nn.TransformerDecoderLayer(d_model=64, nhead=1, dim_feedforward=64, activation='gelu').to(
            device='cuda')
        transformer_decoder_4 = nn.TransformerDecoder(decoder_layer_4, num_layers=1, norm=None).to(device='cuda')
        memory_4 = x4r_output
        target_4 = x4r_output
        x4r_decoder_out = transformer_decoder_4(target_4, memory_4)

        decoder_layer_5 = nn.TransformerDecoderLayer(d_model=64, nhead=1, dim_feedforward=64, activation='gelu').to(
            device='cuda')
        transformer_decoder_5 = nn.TransformerDecoder(decoder_layer_5, num_layers=1, norm=None).to(device='cuda')
        memory_5 = x5r_output
        target_5 = x5r_output
        x5r_decoder_out = transformer_decoder_5(target_5, memory_5)

        decoder_layer_6 = nn.TransformerDecoderLayer(d_model=64, nhead=1, dim_feedforward=64, activation='gelu').to(
            device='cuda')
        transformer_decoder_6 = nn.TransformerDecoder(decoder_layer_6, num_layers=1, norm=None).to(device='cuda')
        memory_6 = x6r_output
        target_6 = x6r_output
        x6r_decoder_out = transformer_decoder_6(target_6, memory_6)

        decoder_layer_7 = nn.TransformerDecoderLayer(d_model=64, nhead=1, dim_feedforward=64, activation='gelu').to(
            device='cuda')
        transformer_decoder_7 = nn.TransformerDecoder(decoder_layer_7, num_layers=1, norm=None).to(device='cuda')
        memory_7 = x7r_output
        target_7 = x7r_output
        x7r_decoder_out = transformer_decoder_7(target_7, memory_7)

        decoder_layer_8 = nn.TransformerDecoderLayer(d_model=64, nhead=1, dim_feedforward=64, activation='gelu').to(
            device='cuda')
        transformer_decoder_8 = nn.TransformerDecoder(decoder_layer_8, num_layers=1, norm=None).to(device='cuda')
        memory_8 = x8r_output
        target_8 = x8r_output
        x8r_decoder_out = transformer_decoder_8(target_8, memory_8)



        # 提取中间像素信息

        # x1r_output_central = x1r_output[0, :, :] + x1r_output[13,:,:]
        # x1r_output_clstoken = x1r_output[0, :, :]
        # x1r_output_centraltoken = x1r_output[13, :, :]
        x1r_output_centraltoken = x1r_decoder_out[13, :, :]
        x1r_output_meantoken = reduce(x1r_output, 'x b c -> b c', reduction='mean')
        # print('1111',x1r_output_central.shape) #(b,c) SA(100, 204)\

        # x2r_output_central = x2r_output[0, :, :] + x2r_output[13,:,:]
        # x2r_output_clstoken = x2r_output[0, :, :]
        # x2r_output_centraltoken = x2r_output[13, :, :]
        x2r_output_centraltoken = x2r_decoder_out[13, :, :]
        x2r_output_meantoken = reduce(x2r_output, 'x b c -> b c', reduction='mean')

        # x3r_output_central = x3r_output[0, :, :] + x3r_output[13,:,:]
        # x3r_output_clstoken = x3r_output[0, :, :]
        # x3r_output_centraltoken = x3r_output[13, :, :]
        x3r_output_centraltoken = x3r_decoder_out[13, :, :]
        x3r_output_meantoken = reduce(x3r_output, 'x b c -> b c', reduction='mean')

        # x4r_output_central = x4r_output[0, :, :] + x4r_output[13,:,:]
        # x4r_output_clstoken = x4r_output[0, :, :]
        # x4r_output_centraltoken = x4r_output[13, :, :]
        x4r_output_centraltoken = x4r_decoder_out[13, :, :]
        x4r_output_meantoken = reduce(x4r_output, 'x b c -> b c', reduction='mean')

        # x5r_output_central = x5r_output[0, :, :] + x5r_output[13,:,:]
        # x5r_output_clstoken = x5r_output[0, :, :]
        # x5r_output_centraltoken = x5r_output[13, :, :]
        x5r_output_centraltoken = x5r_decoder_out[13, :, :]
        x5r_output_meantoken = reduce(x5r_output, 'x b c -> b c', reduction='mean')

        # x6r_output_central = x6r_output[0, :, :] + x6r_output[13,:,:]
        # x6r_output_clstoken = x6r_output[0, :, :]
        # x6r_output_centraltoken = x6r_output[13, :, :]
        x6r_output_centraltoken = x6r_decoder_out[13, :, :]
        x6r_output_meantoken = reduce(x6r_output, 'x b c -> b c', reduction='mean')

        # x7r_output_central = x7r_output[0, :, :] + x7r_output[13,:,:]
        # x7r_output_clstoken = x7r_output[0, :, :]
        # x7r_output_centraltoken = x7r_output[13, :, :]
        x7r_output_centraltoken = x7r_decoder_out[13, :, :]
        x7r_output_meantoken = reduce(x7r_output, 'x b c -> b c', reduction='mean')

        # x8r_output_central = x8r_output[0, :, :] + x8r_output[13,:,:]
        # x8r_output_clstoken = x8r_output[0, :, :]
        # x8r_output_centraltoken = x8r_output[13, :, :]
        x8r_output_centraltoken = x8r_decoder_out[13, :, :]
        x8r_output_meantoken = reduce(x8r_output, 'x b c -> b c', reduction='mean')

        # 扩展维度准备合并
        # x1r_output_centraltoken = rearrange(x1r_output_centraltoken, 'b c -> () b c')
        # x2r_output_centraltoken = rearrange(x2r_output_centraltoken, 'b c -> () b c')
        # x3r_output_centraltoken = rearrange(x3r_output_centraltoken, 'b c -> () b c')
        # x4r_output_centraltoken = rearrange(x4r_output_centraltoken, 'b c -> () b c')
        # x5r_output_centraltoken = rearrange(x5r_output_centraltoken, 'b c -> () b c')
        # x6r_output_centraltoken = rearrange(x6r_output_centraltoken, 'b c -> () b c')
        # x7r_output_centraltoken = rearrange(x7r_output_centraltoken, 'b c -> () b c')
        # x8r_output_centraltoken = rearrange(x8r_output_centraltoken, 'b c -> () b c')
        # print('x1r_output_centraltoken', x1r_output_centraltoken.shape)
        #
        # x1r_output_meantoken = rearrange(x1r_output_meantoken, 'b c -> () b c')
        # x2r_output_meantoken = rearrange(x2r_output_meantoken, 'b c -> () b c')
        # x3r_output_meantoken = rearrange(x3r_output_meantoken, 'b c -> () b c')
        # x4r_output_meantoken = rearrange(x4r_output_meantoken, 'b c -> () b c')
        # x5r_output_meantoken = rearrange(x5r_output_meantoken, 'b c -> () b c')
        # x6r_output_meantoken = rearrange(x6r_output_meantoken, 'b c -> () b c')
        # x7r_output_meantoken = rearrange(x7r_output_meantoken, 'b c -> () b c')
        # x8r_output_meantoken = rearrange(x8r_output_meantoken, 'b c -> () b c')
        # print('x1r_output_meantoken', x1r_output_meantoken.shape)

        '只用一个扫描 or 2 or 4 '
        # x1r_output_conca = torch.cat([x1r_output_meantoken, x1r_output_centraltoken], dim=1)
        # x7r_output_conca = torch.cat([x7r_output_meantoken, x7r_output_centraltoken], dim=1)
        # x2r_output_conca = torch.cat([x2r_output_meantoken, x2r_output_centraltoken], dim=1)
        # x8r_output_conca = torch.cat([x8r_output_meantoken, x8r_output_centraltoken], dim=1)
        # x3r_output_conca = torch.cat([x3r_output_meantoken, x3r_output_centraltoken], dim=1)
        # x4r_output_conca = torch.cat([x4r_output_meantoken, x4r_output_centraltoken], dim=1)
        # x5r_output_conca = torch.cat([x5r_output_meantoken, x5r_output_centraltoken], dim=1)
        # x6r_output_conca = torch.cat([x6r_output_meantoken, x6r_output_centraltoken], dim=1)

        x1r_output_conca = repeat(x1r_output_centraltoken, 'b c -> b c ()')
        x7r_output_conca = repeat(x7r_output_centraltoken, 'b c -> b c ()')
        x2r_output_conca = repeat(x2r_output_centraltoken, 'b c -> b c ()')
        x8r_output_conca = repeat(x8r_output_centraltoken, 'b c -> b c ()')
        x3r_output_conca = repeat(x3r_output_centraltoken, 'b c -> b c ()')
        x4r_output_conca = repeat(x4r_output_centraltoken, 'b c -> b c ()')
        x5r_output_conca = repeat(x5r_output_centraltoken, 'b c -> b c ()')
        x6r_output_conca = repeat(x6r_output_centraltoken, 'b c -> b c ()')

        preds_onedirection = torch.cat(
            [x1r_output_conca, x7r_output_conca, x2r_output_conca, x8r_output_conca, x3r_output_conca, x5r_output_conca,
             x4r_output_conca, x6r_output_conca], dim=2)  # （b c x)

        # # x1r = self.gru_2_1(x1r)[0]
        # print('x1r', x1r.shape)  #(x,b,c) (25,100,64)
        # x1r_laststep = x1r[-1]
        # x1r_laststep = self.relu(x1r_laststep)
        # x1r_laststep = torch.unsqueeze(x1r_laststep, dim=0)

        # print('```',preds_onedirection.shape)
        #

        preds_onedirection = preds_onedirection.view(preds_onedirection.size(0), -1)
        preds_onedirection = self.transformer_bn_scheme1(preds_onedirection)
        preds_onedirection = self.relu(preds_onedirection)
        preds_onedirection = self.dropout(preds_onedirection)
        preds_onedirection = self.fc_transformer_scheme1(preds_onedirection)

        "用LSTM"
        # h0_x1r = torch.zeros(1, x1r.size(1), 64).to(device='cuda')
        # c0_x1r = torch.zeros(1, x1r.size(1), 64).to(device="cuda")
        # x1r, (hn_x1r, cn_x1r) = self.lstm_2_1(x1r, (h0_x1r, c0_x1r))
        # # x1r = self.gru_2_1(x1r)[0]
        # print('x1r', x1r.shape)  #(x,b,c) (25,100,64)
        # x1r_laststep = x1r[-1]
        # x1r_laststep = self.relu(x1r_laststep)
        # x1r_laststep = torch.unsqueeze(x1r_laststep, dim=0)
        #
        # h0_x2r = torch.zeros(1, x2r.size(1), 64).to(device='cuda')
        # c0_x2r = torch.zeros(1, x2r.size(1), 64).to(device="cuda")
        # x2r, (hn_x2r, cn_x2r) = self.lstm_2_2(x2r, (h0_x2r, c0_x2r))
        # # x2r = self.gru_2_2(x2r)[0] #把x1r经过RNN的值，作为x2r的输入
        # x2r_laststep = x2r[-1]
        # x2r_laststep = self.relu(x2r_laststep)
        # x2r_laststep = torch.unsqueeze(x2r_laststep, dim=0)
        #
        # h0_x3r = torch.zeros(1, x3r.size(1), 64).to(device='cuda')
        # c0_x3r = torch.zeros(1, x3r.size(1), 64).to(device="cuda")
        # x3r, (hn_x3r, cn_x3r) = self.lstm_2_3(x3r, (h0_x3r, c0_x3r))
        # # x3r = self.gru_2_3(x3r)[0]
        # x3r_laststep = x3r[-1]
        # x3r_laststep = self.relu(x3r_laststep)
        # x3r_laststep = torch.unsqueeze(x3r_laststep, dim=0)
        #
        # h0_x4r = torch.zeros(1, x4r.size(1), 64).to(device='cuda')
        # c0_x4r = torch.zeros(1, x4r.size(1), 64).to(device="cuda")
        # x4r, (hn_x4r, cn_x4r) = self.lstm_2_4(x4r, (h0_x4r, c0_x4r))
        # # x4r = self.gru_2_4(x4r)[0]
        # x4r_laststep = x4r[-1]
        # x4r_laststep = self.relu(x4r_laststep)
        # x4r_laststep = torch.unsqueeze(x4r_laststep, dim=0)
        #
        # h0_x5r = torch.zeros(1, x5r.size(1), 64).to(device='cuda')
        # c0_x5r = torch.zeros(1, x5r.size(1), 64).to(device="cuda")
        # x5r, (hn_x5r, cn_x5r) = self.lstm_2_5(x5r, (h0_x5r, c0_x5r))
        # # x5r = self.gru_2_5(x5r)[0]
        # x5r_laststep = x5r[-1]
        # x5r_laststep = self.relu(x5r_laststep)
        # x5r_laststep = torch.unsqueeze(x5r_laststep, dim=0)
        #
        # h0_x6r = torch.zeros(1, x6r.size(1), 64).to(device='cuda')
        # c0_x6r = torch.zeros(1, x6r.size(1), 64).to(device="cuda")
        # x6r, (hn_x6r, cn_x6r) = self.lstm_2_6(x6r, (h0_x6r, c0_x6r))
        # # x6r = self.gru_2_6(x6r)[0]
        # x6r_laststep = x6r[-1]
        # x6r_laststep = self.relu(x6r_laststep)
        # x6r_laststep = torch.unsqueeze(x6r_laststep, dim=0)
        #
        # h0_x7r = torch.zeros(1, x7r.size(1), 64).to(device='cuda')
        # c0_x7r = torch.zeros(1, x7r.size(1), 64).to(device="cuda")
        # x7r, (hn_x7r, cn_x7r) = self.lstm_2_7(x7r, (h0_x7r, c0_x7r))
        # # x7r = self.gru_2_7(x7r)[0]
        # x7r_laststep = x7r[-1]
        # x7r_laststep = self.relu(x7r_laststep)
        # x7r_laststep = torch.unsqueeze(x7r_laststep, dim=0)
        #
        # h0_x8r = torch.zeros(1, x8r.size(1), 64).to(device='cuda')
        # c0_x8r = torch.zeros(1, x8r.size(1), 64).to(device="cuda")
        # x8r, (hn_x8r, cn_x8r) = self.lstm_2_8(x8r, (h0_x8r, c0_x8r))
        # # x8r = self.gru_2_8(x8r)[0]
        # x8r_laststep = x8r[-1]
        # x8r_laststep = self.relu(x8r_laststep)
        # x8r_laststep = torch.unsqueeze(x8r_laststep, dim=0)
        # print('x8r_last',x8r_laststep.shape)

        "scheme 1"
        # x_strategy_FLC = torch.cat([x1r_output_central,x2r_output_central,x3r_output_central,x4r_output_central,x5r_output_central,x6r_output_central,x7r_output_central,x8r_output_central],dim=0)
        # print('x_strategy_FLC', x_strategy_FLC.shape) # (x, b, c) (8 , batch, 64)
        # x_strategy_FLC = rearrange(x_strategy_FLC, 'x b c -> b x c')
        # x_strategy_FLC += self.pos_embedding_conca_scheme1[:, :(x_strategy_FLC.size(1))]
        # x_strategy_FLC = rearrange(x_strategy_FLC, 'b x c -> x b c')
        # # 设置transformer的参数给8个direction
        # encoder_layer_conca = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=1, dim_feedforward=32, activation='gelu').to(
        #     device='cuda')
        # transformer_encoder_conca = nn.TransformerEncoder(encoder_layer_conca, num_layers=1, norm=None).to(device='cuda')
        #
        # x_strategy_FLC_output = transformer_encoder_conca(x_strategy_FLC)
        # x_strategy_FLC_output = self.denselayer_scheme1(x_strategy_FLC_output)
        # x_strategy_FLC_output = reduce(x_strategy_FLC_output, 'x b c -> 1 b c', reduction='mean')
        # x_strategy_FLC_output = self.relu(x_strategy_FLC_output)
        # x_strategy_FLC_output = x_strategy_FLC_output.view(x_strategy_FLC_output.size(1), -1)
        # x_strategy_FLC_output = self.transformer_bn_scheme1(x_strategy_FLC_output)
        # x_strategy_FLC_output = self.dropout(x_strategy_FLC_output)
        # preds_scheme1 = self.fc_transformer_scheme1(x_strategy_FLC_output)

        "scheme 2"
        # x1_7r = torch.cat([x1r_output_conca, x7r_output_conca], dim=1 )
        # x2_8r = torch.cat([x2r_output_conca, x8r_output_conca], dim=1 )
        # x3_5r = torch.cat([x3r_output_conca, x5r_output_conca], dim=1 )
        # x4_6r = torch.cat([x4r_output_conca, x6r_output_conca], dim=1 )
        # x_strategy_FLC = torch.cat([x1_7r, x2_8r, x3_5r, x4_6r], dim=2)
        # x_strategy_FLC = rearrange(x_strategy_FLC, 'b c x -> b x c')
        # print('...', x_strategy_FLC.shape)
        # cls_tokens_FLC = repeat(self.cls_token_FLC, '() n c -> b n c', b=b)
        # x_strategy_FLC = torch.cat((cls_tokens_FLC, x_strategy_FLC), dim=1)
        # x_strategy_FLC += self.pos_embedding_conca_scheme2[:, :(x_strategy_FLC.size(1) + 1 )]
        # x_strategy_FLC = rearrange(x_strategy_FLC, 'b x c -> x b c')
        # # 设置transformer的参数给4对directions
        # encoder_layer_conca = nn.TransformerEncoderLayer(d_model=self.embed_dim * 4, nhead=1, dim_feedforward=64,
        #                                                  activation='gelu').to(device='cuda')
        # transformer_encoder_conca = nn.TransformerEncoder(encoder_layer_conca, num_layers=1, norm=None).to(device='cuda')
        # x_strategy_FLC_output = transformer_encoder_conca(x_strategy_FLC)
        # x_strategy_FLC_output = self.denselayer_scheme2(x_strategy_FLC_output)
        #
        # x_strategy_FLC_output_clstoken = x_strategy_FLC_output[0,:,:]
        # x_strategy_FLC_output_meantoken = (x_strategy_FLC_output[1,:,:] + x_strategy_FLC_output[2,:,:] + x_strategy_FLC_output[3,:,:] + x_strategy_FLC_output[4,:,:])
        # # x_strategy_FLC_output = reduce(x_strategy_FLC_output, 'x b c -> 1 b c', reduction='mean')
        #
        # x_strategy_FLC_output = self.relu(x_strategy_FLC_output_meantoken)
        # x_strategy_FLC_output = x_strategy_FLC_output.view(x_strategy_FLC_output.size(0), -1)
        # x_strategy_FLC_output = self.transformer_bn_scheme2(x_strategy_FLC_output)
        # x_strategy_FLC_output = self.dropout(x_strategy_FLC_output)
        # preds_scheme2 = self.fc_transformer_scheme2(x_strategy_FLC_output)

        return preds_onedirection


class zhou_dual_LSTM_Trans(nn.Module):

    @staticmethod
    def weight_init(m):
        # All weight matrices in our RNN and bias vectors are initialized with a uniform distribution, and the values of these weight matrices and bias vectors are initialized in the range [−0.1,0.1]
        if isinstance(m, (nn.Linear, nn.GRU, nn.Conv3d, nn.LSTM, nn.TransformerEncoderLayer, nn.TransformerEncoder)):
            init.kaiming_uniform_(m.weight.data)
            init.kaiming_uniform_(m.bias.data)

    def __init__(self, input_channels, n_classes, patch_size=5, embed_dim=64):
        # The proposed network model uses a single recurrent layer that adopts our modified GRUs of size 64 with sigmoid gate activation and PRetanh activation functions for hidden representations
        super(zhou_dual_LSTM_Trans, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        'do not use self.n_classes = n_classes'
        self.lstm_trans = nn.LSTM(4, 65 * 4, 1, bidirectional=False)
        self.lstm_2_1 = nn.LSTM(self.embed_dim +25, self.embed_dim +25, 1, bidirectional=False)
        self.lstm_2_2 = nn.LSTM(self.embed_dim +25, self.embed_dim +25, 1, bidirectional=False)
        self.lstm_2_3 = nn.LSTM(self.embed_dim +25, self.embed_dim +25, 1, bidirectional=False)
        self.lstm_2_4 = nn.LSTM(self.embed_dim +25, self.embed_dim +25, 1, bidirectional=False)
        self.lstm_2_5 = nn.LSTM(self.embed_dim +25, self.embed_dim +25, 1, bidirectional=False)
        self.lstm_2_6 = nn.LSTM(self.embed_dim +25, self.embed_dim +25, 1, bidirectional=False)
        self.lstm_2_7 = nn.LSTM(self.embed_dim +25, self.embed_dim +25, 1, bidirectional=False)
        self.lstm_2_8 = nn.LSTM(self.embed_dim +25, self.embed_dim +25, 1, bidirectional=False)
        # self.model2 = zhou_slidingLSTM_Trans(input_channels,n_classes,patch_size,embed_dim)
        self.cls_token_1 = nn.Parameter(torch.randn(1, 25, 1))
        self.cls_token_2 = nn.Parameter(torch.randn(1, 25, 1))
        self.cls_token_3 = nn.Parameter(torch.randn(1, 25, 1))
        self.cls_token_4 = nn.Parameter(torch.randn(1, 25, 1))
        self.cls_token_5 = nn.Parameter(torch.randn(1, 25, 1))
        self.cls_token_6 = nn.Parameter(torch.randn(1, 25, 1))
        self.cls_token_7 = nn.Parameter(torch.randn(1, 25, 1))
        self.cls_token_8 = nn.Parameter(torch.randn(1, 25, 1))
        self.pos_embedding_1 = nn.Parameter(torch.randn(1, 25, embed_dim))
        self.pos_embedding_2 = nn.Parameter(torch.randn(1, 25 + 1, embed_dim))
        self.pos_embedding_3 = nn.Parameter(torch.randn(1, 25 + 1, embed_dim))
        self.pos_embedding_4 = nn.Parameter(torch.randn(1, 25 + 1, embed_dim))
        self.pos_embedding_5 = nn.Parameter(torch.randn(1, 25 + 1, embed_dim))
        self.pos_embedding_6 = nn.Parameter(torch.randn(1, 25 + 1, embed_dim))
        self.pos_embedding_7 = nn.Parameter(torch.randn(1, 25 + 1, embed_dim))
        self.pos_embedding_8 = nn.Parameter(torch.randn(1, 25 + 1, embed_dim))
        self.pos_embedding_conca_scheme1 = nn.Parameter(torch.randn(1, 8, embed_dim))
        self.pos_embedding_conca_scheme2 = nn.Parameter(torch.randn(1, 4 + 1, embed_dim * 4))
        self.cls_token_FLC = nn.Parameter(torch.randn(1, 1, embed_dim * 4))
        self.lstm_4 = nn.LSTM(patch_size ** 2, 64, 1)
        self.transformer = nn.Transformer(nhead=16, num_encoder_layers=12)
        self.conv2d_1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=2, stride=1)
        self.dpe = nn.Conv2d(in_channels=input_channels, out_channels=1, kernel_size=1)
        self.dpe_2 = nn.Conv2d(in_channels=input_channels, out_channels=25, kernel_size=1)
        self.depth_conv_1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, padding=1,
                                      groups=input_channels)
        self.point_conv_1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1)
        self.depth_conv_2 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3)
        self.point_conv_2 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3)
        self.conv2d_2 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=2, stride=1)
        self.conv2d_3 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, stride=1)
        self.conv2d_4 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1)
        self.gru_bn = nn.BatchNorm1d(patch_size ** 2 * input_channels)
        self.lstm_bn_1 = nn.BatchNorm1d((64) * 1)
        self.lstm_bn_1_2 = nn.BatchNorm1d((64) * (patch_size) ** 2)
        self.lstm_bn_2 = nn.BatchNorm1d((64) * 8)
        self.lstm_bn_2_2 = nn.BatchNorm1d((64) * 8 * patch_size ** 2)
        self.gru_bn_2 = nn.BatchNorm1d((patch_size ** 2) * (patch_size ** 2))
        self.gru_bn_3 = nn.BatchNorm1d((patch_size ** 2) * (patch_size ** 2) * 2)
        self.gru_bn_4 = nn.BatchNorm1d(8 * 64)
        self.gru_bn_laststep = nn.BatchNorm1d(64)
        self.transformer_bn_scheme1 = nn.BatchNorm1d((embed_dim +25) * 8)
        self.transformer_bn_scheme2 = nn.BatchNorm1d(embed_dim * 2)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(67600, n_classes)
        self.bn = nn.BatchNorm1d(67600)
        self.lstm_fc_1 = nn.Linear(64 * 1, n_classes)
        self.lstm_fc_1_2 = nn.Linear(64 * (patch_size ** 2), n_classes)
        self.lstm_fc_2 = nn.Linear(64 * 8, n_classes)
        self.lstm_fc_2_2 = nn.Linear(64 * 8 * patch_size ** 2, n_classes)
        self.fc_2 = nn.Linear((patch_size ** 2) * (patch_size ** 2), n_classes)
        self.fc_3 = nn.Linear((patch_size ** 2) * (patch_size ** 2) * 2, n_classes)
        self.fc_4 = nn.Linear(8 * 64, n_classes)
        self.layernorm = nn.LayerNorm(embed_dim)
        self.denselayer = nn.Linear(embed_dim +25, embed_dim +25)
        self.denselayer_scheme1 = nn.Linear(embed_dim, embed_dim)
        self.denselayer_scheme2 = nn.Linear(embed_dim * 4, embed_dim * 2)
        self.fc_laststep = nn.Linear(64, n_classes)
        self.fc_transformer_scheme1 = nn.Linear((embed_dim +25) * 8, n_classes)
        self.fc_transformer_scheme2 = nn.Linear(embed_dim * 2, n_classes)
        self.linearembedding = nn.Linear(in_features=input_channels, out_features=embed_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):  # 初始是第1方向
        print('x.shape1', x.shape)
        x = x.squeeze(1)  # b,c,h w

        x_plot = rearrange(x, 'b c h w  -> b (h w) c')

        # for i in range(x_plot.size(1)):
        #     plt.plot(x_plot[1,i,:].cpu().detach().numpy())
        #     plt.xticks(fontsize=20)
        #     plt.yticks(fontsize=20)
        #     plt.grid(visible=True)
        # # plt.show()
        #
        x_label = x_plot[:,:,0]
        x_label = rearrange(x_label,'b n -> (n b)').cpu().detach().numpy()
        x_label[x_label == 0] = np.nan
        # predict = self.model2(x)

        '这一节主要是为了生成 dynamic positional embedding （dpe） 和 multi-scanning的dpe'
        # x1 = x  # 用于保存原数据
        # x2 = x  # 用于生成dpe
        # dpe = self.dpe(x2)
        # print(dpe.dtype)
        # print('dpe', dpe)  # (b 1 5 5)
        #
        # # 试一试one-hot
        # # dpe_onehot = one_hot_extend(dpe)
        # #
        # # onehotencoder = OneHotEncoder(categories='auto')
        # # for i in range(dpe_onehot.size(0)):
        # #     i_th = dpe_onehot[i,:]
        # #     i_th = i_th.unsqueeze(0)
        # #     onehotoutput = onehotencoder.fit_transform(i_th.cpu().detach().numpy())
        # #     print(onehotoutput.toarray())
        # # i = i + 1
        # #
        # #
        #
        # # multi-dpe (把dpe也换成multiscanning)
        # # 生成1和7
        # dpe1_0 = dpe[:, :, 0, :]
        # dpe1_1 = dpe[:, :, 1, :]
        # dpe1_2 = dpe[:, :, 2, :]
        # dpe1_3 = dpe[:, :, 3, :]
        # dpe1_4 = dpe[:, :, 4, :]
        # dpe1_1f = torch.flip(dpe1_1, [2])
        # dpe1_3f = torch.flip(dpe1_3, [2])
        # # direction_1 = torch.cat([x1_0, x1_1f, x1_2], dim=2)
        # dpe_1 = torch.cat([dpe1_0, dpe1_1f, dpe1_2, dpe1_3f, dpe1_4], dim=2)
        # print('dpe_1', dpe_1.shape)  # （100, 1 ,25）(b c x)
        # dpe_7 = torch.flip(dpe_1, [2])
        # dpe_1 = repeat(dpe_1, 'b 1 x -> b x 25')
        # dpe_7 = repeat(dpe_7, 'b 1 x -> b x 25')
        # print('dpe_1', dpe_1.shape)  # （100, 1 ,25）(b c x)
        # # plt.subplot(241)
        # # plt.imshow(dpe_1[1, :, :].cpu().detach().numpy())
        # # plt.subplot(247)
        # # plt.imshow(dpe_7[1, :, :].cpu().detach().numpy())
        # # plt.show()
        #
        # # 生成第2和8
        # dpe2_0 = dpe[:, :, :, 0]
        # dpe2_1 = dpe[:, :, :, 1]
        # dpe2_2 = dpe[:, :, :, 2]
        # dpe2_3 = dpe[:, :, :, 3]
        # dpe2_4 = dpe[:, :, :, 4]
        # dpe2_1f = torch.flip(dpe2_1, [2])
        # dpe2_3f = torch.flip(dpe2_3, [2])
        # # direction_2 = torch.cat([x2_0, x2_1f, x2_2], dim=2)
        # dpe_2 = torch.cat([dpe2_0, dpe2_1f, dpe2_2, dpe2_3f, dpe2_4], dim=2)
        # dpe_8 = torch.flip(dpe_2, [2])
        # dpe_2 = repeat(dpe_2, 'b 1 x -> b x 25')
        # dpe_8 = repeat(dpe_8, 'b 1 x -> b x 25')
        # # plt.subplot(242)
        # # plt.imshow(dpe_2[1, :, :].cpu().detach().numpy())
        # # plt.subplot(248)
        # # plt.imshow(dpe_8[1, :, :].cpu().detach().numpy())
        #
        # # 生成3和5
        # dpe3_0 = dpe[:, :, 0, :]
        # dpe3_1 = dpe[:, :, 1, :]
        # dpe3_2 = dpe[:, :, 2, :]
        # dpe3_3 = dpe[:, :, 3, :]
        # dpe3_4 = dpe[:, :, 4, :]
        # dpe3_0f = torch.flip(dpe3_0, [2])
        # dpe3_2f = torch.flip(dpe3_2, [2])
        # dpe3_4f = torch.flip(dpe3_4, [2])
        # # direction_3 = torch.cat([x3_0f, x3_1, x3_2f], dim=2)
        # dpe_3 = torch.cat([dpe3_0f, dpe3_1, dpe3_2f, dpe3_3, dpe3_4f], dim=2)
        # dpe_5 = torch.flip(dpe_3, [2])
        # dpe_3 = repeat(dpe_3, 'b 1 x -> b x 25')
        # dpe_5 = repeat(dpe_5, 'b 1 x -> b x 25')
        # # plt.subplot(243)
        # # plt.imshow(dpe_3[1, :, :].cpu().detach().numpy())
        # # plt.subplot(245)
        # # plt.imshow(dpe_5[1, :, :].cpu().detach().numpy())
        #
        # # 生成4和6
        # dpe4_0 = dpe[:, :, :, 0]
        # dpe4_1 = dpe[:, :, :, 1]
        # dpe4_2 = dpe[:, :, :, 2]
        # dpe4_3 = dpe[:, :, :, 3]
        # dpe4_4 = dpe[:, :, :, 4]
        # dpe4_1f = torch.flip(dpe4_1, [2])
        # dpe4_3f = torch.flip(dpe4_3, [2])
        # # direction_4 = torch.cat([x4_2, x4_1f, x4_0], dim=2)
        # dpe_4 = torch.cat([dpe4_4, dpe4_3f, dpe4_2, dpe4_1f, dpe4_0], dim=2)
        # # print('d4', direction_4.shape)
        # dpe_6 = torch.flip(dpe_4, [2])
        # dpe_4 = repeat(dpe_4, 'b 1 x -> b x 25')
        # dpe_6 = repeat(dpe_6, 'b 1 x -> b x 25')
        # # plt.subplot(244)
        # # plt.imshow(dpe_4[1, :, :].cpu().detach().numpy())
        # # plt.subplot(246)
        # # plt.imshow(dpe_6[1, :, :].cpu().detach().numpy())
        # # plt.show()
        #
        # # dpe = reduce(dpe, 'b 1 h w -> b h w', reduction='mean')
        # # dpe = rearrange(dpe, 'b h w -> b (h w)')
        # # dpe = repeat(dpe, 'b x -> b x 25')
        # # print('dpe',dpe.shape)
        # # # dpe = self.sigmoid(dpe)
        # # plt.subplot(212)
        # # plt.imshow(dpe[1,:,:].cpu().detach().numpy())
        # # plt.show()
        # # print(dpe)
        # # xseq = rearrange(x,'b c h w -> b c (h w)')
        # # print(xseq.shape)
        #
        # # ResNet patch_size = 9 for SA PU
        # # x = self.conv2d_1(x)
        # # print('1', x.shape)
        # # x = self.relu(x)
        # # x = self.conv2d_2(x)
        # # print('2', x.shape)
        # # x_res = self.relu(x)
        # # x_res = self.conv2d_3(x_res)
        # # print('3', x.shape) #(ptach size = 6)
        # # x_res = self.relu(x_res)
        # # x_res_res = self.conv2d_4(x_res)
        # # x_res_res = self.relu(x_res_res)
        # # x = x_res + x_res_res
        # # print('4', x.shape)
        # # Depthwise separable convolution
        # # x1 = self.depth_conv_2(x)
        # # x1 = self.point_conv_2(x1)
        # # x = self.depth_conv_1(x)
        # # x = self.point_conv_1(x)
        # # x = self.tanh(x)
        # x = x1 + x


        '这里考虑一下要不要用 depth-wise 和 point-wise 的卷积'
        # x = self.depth_conv_2(x)
        # x = self.point_conv_2(x)

        '这里开始生成multi-scanning的8个方向'
        # 生成第1和7
        x1_0 = x[:, :, 0, :]
        x1_1 = x[:, :, 1, :]
        x1_2 = x[:, :, 2, :]
        x1_3 = x[:, :, 3, :]
        x1_4 = x[:, :, 4, :]
        x1_1f = torch.flip(x1_1, [2])
        x1_3f = torch.flip(x1_3, [2])
        # direction_1 = torch.cat([x1_0, x1_1f, x1_2], dim=2)
        direction_1 = torch.cat([x1_0, x1_1f, x1_2, x1_3f, x1_4], dim=2)
        direction_7 = torch.flip(direction_1, [2])
        print('d1', direction_1.shape)  # b c x 100 103 25


        '这一节主要是为了测试一下 spectral 和 spatial 的 soft attention mask'
        # soft attention mask 速度会变慢
        # direction_1_cdist = rearrange(direction_1, 'b c x -> b x c')
        # dist = torch.cdist(direction_1_cdist,direction_1_cdist,p=2)
        # print('dist',dist.shape)
        # sns.heatmap(dist[1, :, :].cpu().detach().numpy(),cmap='Blues',linewidths=0.5,annot=True,annot_kws={'fontweight':'bold'},square=True)
        # plt.colorbar()
        # plt.show()
        # weight = -dist
        # weight = weight + 1
        # sns.heatmap(weight[1, :, :].cpu().detach().numpy(),cmap='Blues',linewidths=0.5,annot=True,annot_kws={'fontweight':'bold'},square=True)
        # plt.imshow(weight[1, :, :].cpu().detach().numpy(),cmap='blues')
        # plt.colorbar()
        # plt.show()
        # print('d1',direction_1.shape)



        # 生成第2和8
        x2_0 = x[:, :, :, 0]
        x2_1 = x[:, :, :, 1]
        x2_2 = x[:, :, :, 2]
        x2_3 = x[:, :, :, 3]
        x2_4 = x[:, :, :, 4]
        x2_1f = torch.flip(x2_1, [2])
        x2_3f = torch.flip(x2_3, [2])
        # direction_2 = torch.cat([x2_0, x2_1f, x2_2], dim=2)
        direction_2 = torch.cat([x2_0, x2_1f, x2_2, x2_3f, x2_4], dim=2)
        direction_8 = torch.flip(direction_2, [2])

        # 生成3和5
        x3_0 = x[:, :, 0, :]
        x3_1 = x[:, :, 1, :]
        x3_2 = x[:, :, 2, :]
        x3_3 = x[:, :, 3, :]
        x3_4 = x[:, :, 4, :]
        x3_0f = torch.flip(x3_0, [2])
        x3_2f = torch.flip(x3_2, [2])
        x3_4f = torch.flip(x3_4, [2])
        # direction_3 = torch.cat([x3_0f, x3_1, x3_2f], dim=2)
        direction_3 = torch.cat([x3_0f, x3_1, x3_2f, x3_3, x3_4f], dim=2)
        direction_5 = torch.flip(direction_3, [2])

        # 生成4和6
        x4_0 = x[:, :, :, 0]
        x4_1 = x[:, :, :, 1]
        x4_2 = x[:, :, :, 2]
        x4_3 = x[:, :, :, 3]
        x4_4 = x[:, :, :, 4]
        x4_1f = torch.flip(x4_1, [2])
        x4_3f = torch.flip(x4_3, [2])
        # direction_4 = torch.cat([x4_2, x4_1f, x4_0], dim=2)
        direction_4 = torch.cat([x4_4, x4_3f, x4_2, x4_1f, x4_0], dim=2)
        # print('d4', direction_4.shape)
        direction_6 = torch.flip(direction_4, [2])
        # plt.subplot(121)
        # plt.plot(direction_1[1,:,:].cpu().detach().numpy())
        # plt.subplot(122)
        # plt.plot(self.sigmoid(direction_1[1,:,:]).cpu().detach().numpy())
        # plt.show()

        '这里是一系列的plt.show()，主要是plot上面的sequences'
        # plt.xlabel('Band Numbers', fontdict={'size': 35}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 35}, fontweight='bold')
        # plt.plot(direction_1[0, :, 0].cpu().detach().numpy(), label='index 0')
        # plt.plot(direction_1[0, :, 1].cpu().detach().numpy(), label='index 1')
        # plt.plot(direction_1[0, :, 2].cpu().detach().numpy(), label='index 2')
        # plt.plot(direction_1[0, :, 3].cpu().detach().numpy(), label='index 3')
        # plt.plot(direction_1[0, :, 4].cpu().detach().numpy(), label='index 4')
        # plt.plot(direction_1[0, :, 5].cpu().detach().numpy(), label='index 9')
        # plt.plot(direction_1[0, :, 6].cpu().detach().numpy(), label='index 8')
        # plt.plot(direction_1[0, :, 7].cpu().detach().numpy(), label='index 7')
        # plt.plot(direction_1[0, :, 8].cpu().detach().numpy(), label='index 6')
        # plt.plot(direction_1[0, :, 9].cpu().detach().numpy(), label='index 5')
        # plt.plot(direction_1[0, :, 10].cpu().detach().numpy(), label='index 10')
        # plt.plot(direction_1[0, :, 11].cpu().detach().numpy(), label='index 11')
        # plt.plot(direction_1[0, :, 12].cpu().detach().numpy(), label='index 12', linewidth=5, linestyle='-.', color = 'red' )
        # plt.plot(direction_1[0, :, 13].cpu().detach().numpy(), label='index 13')
        # plt.plot(direction_1[0, :, 14].cpu().detach().numpy(), label='index 14')
        # plt.plot(direction_1[0, :, 15].cpu().detach().numpy(), label='index 19')
        # plt.plot(direction_1[0, :, 16].cpu().detach().numpy(), label='index 18')
        # plt.plot(direction_1[0, :, 17].cpu().detach().numpy(), label='index 17')
        # plt.plot(direction_1[0, :, 18].cpu().detach().numpy(), label='index 16')
        # plt.plot(direction_1[0, :, 19].cpu().detach().numpy(), label='index 15')
        # plt.plot(direction_1[0, :, 20].cpu().detach().numpy(), label='index 20')
        # plt.plot(direction_1[0, :, 21].cpu().detach().numpy(), label='index 21')
        # plt.plot(direction_1[0, :, 22].cpu().detach().numpy(), label='index 22')
        # plt.plot(direction_1[0, :, 23].cpu().detach().numpy(), label='index 23')
        # plt.plot(direction_1[0, :, 24].cpu().detach().numpy(), label='index 24')
        # plt.legend(prop={'size': 15}, fontsize='large', loc=[1, 0])
        # plt.title('Spectral signatures in a patch', fontdict={'size': 35}, fontweight='bold')
        # plt.grid(linewidth = 1.5)
        # plt.xticks(fontsize=25)
        # plt.yticks(fontsize=25)
        # plt.show()
        # plt.subplot(122)
        # plt.xlabel('Band Numbers', fontdict={'size': 35}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 35}, fontweight='bold')
        # plt.plot(direction_2[0, :, 0].cpu().detach().numpy(), label='(0,0),0')
        # plt.plot(direction_2[0, :, 1].cpu().detach().numpy(), label='(1,0),5')
        # plt.plot(direction_2[0, :, 2].cpu().detach().numpy(), label='(2,0),10')
        # plt.plot(direction_2[0, :, 3].cpu().detach().numpy(), label='(3,0),15')
        # plt.plot(direction_2[0, :, 4].cpu().detach().numpy(), label='(4,0),20')
        # plt.plot(direction_2[0, :, 5].cpu().detach().numpy(), label='(4,1),21')
        # plt.plot(direction_2[0, :, 6].cpu().detach().numpy(), label='(3,1),16')
        # plt.plot(direction_2[0, :, 7].cpu().detach().numpy(), label='(2,1),11')
        # plt.plot(direction_2[0, :, 8].cpu().detach().numpy(), label='(1,1),6')
        # plt.plot(direction_2[0, :, 9].cpu().detach().numpy(), label='(0,1),1')
        # plt.plot(direction_2[0, :, 10].cpu().detach().numpy(), label='(0,2),2')
        # plt.plot(direction_2[0, :, 11].cpu().detach().numpy(), label='(1,2),7')
        # plt.plot(direction_2[0, :, 12].cpu().detach().numpy(), label='(2,2), center', linewidth=3, linestyle='-.')
        # plt.plot(direction_2[0, :, 13].cpu().detach().numpy(), label='(3,2),17')
        # plt.plot(direction_2[0, :, 14].cpu().detach().numpy(), label='(4,2),22')
        # plt.plot(direction_2[0, :, 15].cpu().detach().numpy(), label='(4,3),23')
        # plt.plot(direction_2[0, :, 16].cpu().detach().numpy(), label='(3,3),18')
        # plt.plot(direction_2[0, :, 17].cpu().detach().numpy(), label='(2,3),13')
        # plt.plot(direction_2[0, :, 18].cpu().detach().numpy(), label='(1,3),8')
        # plt.plot(direction_2[0, :, 19].cpu().detach().numpy(), label='(0,3),3', linewidth=5)
        # plt.plot(direction_2[0, :, 20].cpu().detach().numpy(), label='(0,4),4', linewidth=5)
        # plt.plot(direction_2[0, :, 21].cpu().detach().numpy(), label='(1,4),9', linewidth=5)
        # plt.plot(direction_2[0, :, 22].cpu().detach().numpy(), label='(2,4),14')
        # plt.plot(direction_2[0, :, 23].cpu().detach().numpy(), label='(3,4),19')
        # plt.plot(direction_2[0, :, 24].cpu().detach().numpy(), label='(4,4),24')
        # plt.legend(prop={'size': 15}, fontsize='large', loc=[1, 0])
        # plt.title('Spectral signatures in a patch', fontdict={'size': 35}, fontweight='bold')
        # plt.xticks(fontsize=25)
        # plt.yticks(fontsize=25)
        # plt.show()

        # # plt.subplot(332)
        # plt.imshow(direction_1[0, :, (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-1 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(333)
        # plt.imshow(direction_2[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-2 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(334)
        # plt.imshow(direction_3[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-3 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(335)
        # plt.imshow(direction_4[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-4 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(336)
        # plt.imshow(direction_5[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-5 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(337)
        # plt.imshow(direction_6[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-6 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(338)
        # plt.imshow(direction_7[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()ticks(fontsize=20)
        # plt.title('Direction-7 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(339)
        # plt.imshow(direction_8[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-8 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()

        # 换成输入顺序

        x8r = direction_8.permute(2, 0, 1)
        x7r = direction_7.permute(2, 0, 1)
        x6r = direction_6.permute(2, 0, 1)
        x5r = direction_5.permute(2, 0, 1)
        x4r = direction_4.permute(2, 0, 1)
        x3r = direction_3.permute(2, 0, 1)
        x2r = direction_2.permute(2, 0, 1)
        x1r = direction_1.permute(2, 0, 1)
        print('d5.shape', x5r.shape)

        print('x1r', x1r.shape)  # (x,b,c) (25,100,204)

        # linear embedding

        x1r = self.linearembedding(x1r)
        x2r = self.linearembedding(x2r)
        x3r = self.linearembedding(x3r)
        x4r = self.linearembedding(x4r)
        x5r = self.linearembedding(x5r)
        x6r = self.linearembedding(x6r)
        x7r = self.linearembedding(x7r)
        x8r = self.linearembedding(x8r)  # （25,,100 64)

        '第一次TSNE'
        x1r_tsne1 = rearrange(x1r, 'n b c -> (n b) c').cpu().detach().numpy()
        print('x1r_tsne1',x1r_tsne1.shape)
        x1r_tsne_out1 = TSNE(n_components=2,verbose=3).fit_transform(x1r_tsne1)
        print('x1r_tsne_out1',x1r_tsne_out1.shape)
        im1 = sns.jointplot(x1r_tsne_out1[:,0],x1r_tsne_out1[:,1],hue=x_label,palette='Paired',s=5)
        plt.show()


        '这里主要是尝试了一下根据绝对位置的编码 1～length'
        # # position_ids
        seq_len, batch_size, feature_dim = x1r.shape[0], x1r.shape[1], x1r.shape[2]
        # print('...', seq_len)
        # position_ids = torch.arange(0, seq_len).to(device='cuda')
        # print('...', position_ids.shape)
        # position_ids = repeat(position_ids, ' n -> b n 1', b=batch_size)
        # print('...', position_ids.shape)

        '这里主要尝试了one-hot位置的编码，25*25的矩阵，位置上的值为1'
        # one_hot
        onehot_encoding = F.one_hot(torch.arange(0, seq_len), num_classes=25).to(device='cuda')
        print('onehot.shape', onehot_encoding.shape)
        onehot_encoding = repeat(onehot_encoding, 'x y -> b x y', b=batch_size)
        print('onehot.shape', onehot_encoding.shape)  # 100, 25, 25
        # plt.imshow(onehot_encoding[1,:,:].cpu().detach().numpy())
        # plt.show()
        # onehot_encoding_1 = onehot_encoding + (dpe_1)
        # # plt.subplot(241)
        # # plt.imshow(onehot_encoding_1[1, :, :].cpu().detach().numpy())
        # onehot_encoding_2 = onehot_encoding + (dpe_2)
        # # plt.subplot(242)
        # # plt.imshow(onehot_encoding_2[1, :, :].cpu().detach().numpy())
        # onehot_encoding_3 = onehot_encoding + (dpe_3)
        # # plt.subplot(243)
        # # plt.imshow(onehot_encoding_3[1, :, :].cpu().detach().numpy())
        # onehot_encoding_4 = onehot_encoding + (dpe_4)
        # # plt.subplot(244)
        # # plt.imshow(onehot_encoding_4[1, :, :].cpu().detach().numpy())
        # onehot_encoding_5 = onehot_encoding + (dpe_5)
        # # plt.subplot(245)
        # # plt.imshow(onehot_encoding_5[1, :, :].cpu().detach().numpy())
        # onehot_encoding_6 = onehot_encoding + (dpe_6)
        # # plt.subplot(246)
        # # plt.imshow(onehot_encoding_6[1, :, :].cpu().detach().numpy())
        # onehot_encoding_7 = onehot_encoding + (dpe_7)
        # # plt.subplot(247)
        # # plt.imshow(onehot_encoding_7[1, :, :].cpu().detach().numpy())
        # onehot_encoding_8 = onehot_encoding + (dpe_8)
        # # plt.subplot(248)
        # # plt.imshow(onehot_encoding_8[1, :, :].cpu().detach().numpy())
        # # onehot_encoding = self.softmax(onehot_encoding)
        # # print('onehot + dpe', onehot_encoding[1,:,:])
        # # plt.show()

        '这里只是做了一下rearrange和cls_token,无伤大雅，不用管'
        x1r = rearrange(x1r, 'x b c -> b x c')
        b, n, c = x1r.shape
        cls_tokens_x1r = repeat(self.cls_token_1, '() n c -> b n c', b=b)
        # cls_tokens_x1r = self.softmax(cls_tokens_x1r)
        # x1r = x1r * cls_tokens_x1r
        # x1r = torch.cat((cls_tokens_x1r, x1r), dim=2)
        # x1r_2_position_ids = torch.cat((position_ids, x1r * 10), dim=2)
        x1r = torch.cat((onehot_encoding, x1r), dim=2)


        # sin and cos pos
        # p_enc_1d_model = PositionalEncoding1D(64)
        # penc_no_sum = p_enc_1d_model(x1r)
        # print('penc', penc_no_sum.shape)
        # x1r_3_sincos = penc_no_sum + x1r
        # plt.imshow(penc_no_sum[1,:,:].cpu().detach().numpy())
        # plt.show()
        # plt.imshow(x1r_3_sincos[1,:,:].cpu().detach().numpy())
        # plt.show()

        # x1r_s = x1r + self.pos_embedding_1[:, :(x1r.size(1))]

        # plt.subplot(241)
        # plt.imshow(x1r[1, :, :].cpu().detach().numpy())
        # print('onehot.shape + seq', x1r_s.shape)
        # plt.imshow(x1r_2_position_ids[1,:,:].cpu().detach().numpy())
        # plt.colorbar()
        # plt.show()

        x2r = rearrange(x2r, 'x b c -> b x c')
        cls_tokens_x2r = repeat(self.cls_token_2, '() n c -> b n c', b=b)
        # cls_tokens_x2r = self.softmax(cls_tokens_x2r)
        # x2r = x2r * cls_tokens_x2r
        # x2r = torch.cat((cls_tokens_x2r, x2r), dim=2)
        # x2r = torch.cat((position_ids, x2r), dim=2)
        x2r = torch.cat((onehot_encoding, x2r), dim=2)
        # plt.subplot(242)
        # plt.imshow(x2r[1, :, :].cpu().detach().numpy())

        # x2r += self.pos_embedding_2[:, :(x2r.size(1) +1)]

        x3r = rearrange(x3r, 'x b c -> b x c')
        cls_tokens_x3r = repeat(self.cls_token_3, '() n c -> b n c', b=b)
        # cls_tokens_x3r = self.softmax(cls_tokens_x3r)
        # x3r = x3r * cls_tokens_x3r
        # x3r = torch.cat((cls_tokens_x3r, x3r), dim=2)
        # x3r = torch.cat((position_ids, x3r), dim=2)
        x3r = torch.cat((onehot_encoding, x3r), dim=2)
        # plt.subplot(243)
        # plt.imshow(x3r[1, :, :].cpu().detach().numpy())
        # x3r += self.pos_embedding_3[:, :(x3r.size(1)+1)]

        x4r = rearrange(x4r, 'x b c -> b x c')
        cls_tokens_x4r = repeat(self.cls_token_4, '() n c -> b n c', b=b)
        # cls_tokens_x4r = self.softmax(cls_tokens_x4r)
        # x4r = x4r * cls_tokens_x4r
        # x4r = torch.cat((cls_tokens_x4r, x4r), dim=2)
        # x4r = torch.cat((position_ids, x4r), dim=2)
        x4r = torch.cat((onehot_encoding, x4r), dim=2)
        # plt.subplot(244)
        # plt.imshow(x4r[1, :, :].cpu().detach().numpy())
        # x4r += self.pos_embedding_4[:, :(x4r.size(1)+1)]

        x5r = rearrange(x5r, 'x b c -> b x c')
        cls_tokens_x5r = repeat(self.cls_token_5, '() n c -> b n c', b=b)
        # cls_tokens_x5r = self.softmax(cls_tokens_x5r)
        # x5r = x5r * cls_tokens_x5r
        # x5r = torch.cat((cls_tokens_x5r, x5r), dim=2)
        # x5r = torch.cat((position_ids, x5r), dim=2)
        x5r = torch.cat((onehot_encoding, x5r), dim=2)
        # plt.subplot(245)
        # plt.imshow(x5r[1, :, :].cpu().detach().numpy())
        # x5r += self.pos_embedding_5[:, :(x5r.size(1)+1)]

        x6r = rearrange(x6r, 'x b c -> b x c')
        cls_tokens_x6r = repeat(self.cls_token_6, '() n c -> b n c', b=b)
        # cls_tokens_x6r = self.softmax(cls_tokens_x6r)
        # x6r = x6r * cls_tokens_x6r
        # x6r = torch.cat((cls_tokens_x6r, x6r), dim=2)
        # x6r = torch.cat((position_ids, x6r), dim=2)
        x6r = torch.cat((onehot_encoding, x6r), dim=2)
        # plt.subplot(246)
        # plt.imshow(x6r[1, :, :].cpu().detach().numpy())
        # x6r += self.pos_embedding_6[:, :(x6r.size(1)+1)]

        x7r = rearrange(x7r, 'x b c -> b x c')
        cls_tokens_x7r = repeat(self.cls_token_7, '() n c -> b n c', b=b)
        # cls_tokens_x7r = self.softmax(cls_tokens_x7r)
        # x7r = x7r * cls_tokens_x7r
        # x7r = torch.cat((cls_tokens_x7r, x7r), dim=2)
        # x7r = torch.cat((position_ids, x7r), dim=2)
        x7r = torch.cat((onehot_encoding, x7r), dim=2)
        # plt.subplot(247)
        # plt.imshow(x7r[1, :, :].cpu().detach().numpy())
        # x7r += self.pos_embedding_7[:, :(x7r.size(1)+1)]

        x8r = rearrange(x8r, 'x b c -> b x c')
        cls_tokens_x8r = repeat(self.cls_token_8, '() n c -> b n c', b=b)
        # cls_tokens_x8r = self.softmax(cls_tokens_x8r)
        # x8r = x8r * cls_tokens_x8r
        # x8r = torch.cat((cls_tokens_x8r, x8r), dim=2)
        # x8r = torch.cat((position_ids, x8r), dim=2)
        x8r = torch.cat((onehot_encoding, x8r), dim=2)
        # plt.subplot(248)
        # plt.imshow(x8r[1, :, :].cpu().detach().numpy())
        # x8r += self.pos_embedding_8[:, :(x8r.size(1)+1)]
        # plt.show()

        #做个rearrange
        x1r = rearrange(x1r, 'b x c -> x b c')  # (100, 25, 64+25)-->(25, 100, 64+25)， （len, batch, fea_dim）
        x2r = rearrange(x2r, 'b x c -> x b c')
        x3r = rearrange(x3r, 'b x c -> x b c')
        x4r = rearrange(x4r, 'b x c -> x b c')
        x5r = rearrange(x5r, 'b x c -> x b c')
        x6r = rearrange(x6r, 'b x c -> x b c')
        x7r = rearrange(x7r, 'b x c -> x b c')
        x8r = rearrange(x8r, 'b x c -> x b c')

        '尝试用LSTMcell做一个LSTM的进程，并保留所有的cell memory'
        #用LSTM做positional encoding and projection
        # hx = torch.zeros(x1r.size(1), 64)
        # cx = torch.zeros(x1r.size(1), 64)
        # output1 = []
        # output1hx = []
        # output1cx = []
        # for i in range(x1r.size(0)):
        #     hx, cx = self.lstm1(x1r[i,:,:], (hx,cx))
        #     output1.append(hx)
        #     output1hx.append(hx)
        #     output1cx.append(cx)
        # output1 = torch.stack(output1,dim=0)
        # output1hx = torch.stack(output1hx,dim=0)
        # output1cx = torch.stack(output1cx,dim=0)
        # print('hn_size:', output1hx.shape, 'cn_size:', output1cx.shape)
        #
        # plt.subplot(131)
        # sns.heatmap(x1r[:,0,:].cpu().detach().numpy())
        # plt.subplot(132)
        # sns.heatmap(output1hx[:,0,:].cpu().detach().numpy())
        # plt.subplot(133)
        # sns.heatmap(output1cx[:, 0, :].cpu().detach().numpy())
        # plt.show()

        # x1r, hids_1, cells_1 = lstm_cell_memory(x1r,64,64)
        # print('outs:',x1r.shape,'hids:',hids_1.shape,'cells:',cells_1.shape)

        h0_x1r = Variable(torch.zeros(1, x1r.size(1), self.embed_dim +25)).to(device='cuda') #可以试试torch.zeros或者torch.rand
        c0_x1r = Variable(torch.zeros(1, x1r.size(1), self.embed_dim +25)).to(device="cuda") #要不要加requires_grad=True ?
        x1r_lstm, (hn_x1r, cn_x1r) = self.lstm_2_1(x1r, (h0_x1r, c0_x1r))
        var1 = torch.var(x1r_lstm)
        print('var1:',var1)
        print('hn_size:', hn_x1r.shape, 'cn_size:', cn_x1r.shape)
        print('x1r_lstm：', x1r.shape)  #(x,b,c) (25,100,64)
        # plt.subplot(121)
        # sns.heatmap(x1r_lstm[:,1,:].cpu().detach().numpy())
        # x1r_laststep = x1r[-1]
        # x1r_laststep = self.relu(x1r_laststep)
        # x1r_laststep = torch.unsqueeze(x1r_laststep, dim=0)

        '第二次TSNE'
        x1r_tsne2 = rearrange(x1r_lstm, 'n b c -> (n b) c').cpu().detach().numpy()
        print('x1r_tsne1', x1r_tsne2.shape)
        x1r_tsne_out2 = TSNE(n_components=2, verbose=3).fit_transform(x1r_tsne2)

        im2 = sns.jointplot(x1r_tsne_out2[:, 0], x1r_tsne_out2[:, 1], hue=x_label, palette='Paired', s=5)
        plt.show()


        '这一节也主要是为了测试一下 spectral 和 spatial 的 soft attention mask'
        # direction_1_cdist = rearrange(x1r, 'x b c -> b x c')
        # dist = torch.cdist(direction_1_cdist,direction_1_cdist,p=2)
        # print('dist',dist.shape)
        # sns.heatmap(dist[0, :, :].cpu().detach().numpy(),linewidths=0.5,annot=True,annot_kws={'fontweight':'bold'},square=True)
        # plt.title('dist')
        # plt.show()
        # weight = -dist
        # weight = weight + 1
        # sns.heatmap(weight[1, :, :].cpu().detach().numpy(),cmap='Blues',linewidths=0.5,annot=True,annot_kws={'fontweight':'bold'},square=True)
        # plt.title('weight')
        # plt.show()
        # print('d1',direction_1.shape)

        # x1rplot = rearrange(x1r, 'l b c -> c b l')
        # plt.subplot(241)
        # plt.plot(x1rplot[:,0,:].cpu().detach().numpy())

        # x2r, hids_2, cells_2 = lstm_cell_memory(x2r, 64, 64)

        h0_x2r = Variable(torch.zeros(1, x2r.size(1), self.embed_dim +25)).to(device='cuda')
        c0_x2r = Variable(torch.zeros(1, x2r.size(1), self.embed_dim +25)).to(device="cuda")
        x2r_lstm, (hn_x2r, cn_x2r) = self.lstm_2_2(x2r, (h0_x2r, c0_x2r))
        # x2r = self.gru_2_2(x2r)[0] #把x1r经过RNN的值，作为x2r的输入
        # x2r_laststep = x2r[-1]
        # x2r_laststep = self.relu(x2r_laststep)
        # x2r_laststep = torch.unsqueeze(x2r_laststep, dim=0)
        # x2rplot = rearrange(x2r, 'l b c -> c b l')
        # plt.subplot(242)
        # plt.plot(x2rplot[:, 0, :].cpu().detach().numpy())

        # x3r, hids_3, cells_3 = lstm_cell_memory(x3r, 64, 64)

        h0_x3r = Variable(torch.zeros(1, x3r.size(1), self.embed_dim +25)).to(device='cuda')
        c0_x3r = Variable(torch.zeros(1, x3r.size(1), self.embed_dim +25)).to(device="cuda")
        x3r_lstm, (hn_x3r, cn_x3r) = self.lstm_2_3(x3r, (h0_x3r, c0_x3r))
        # x3r = self.gru_2_3(x3r)[0]
        # x3r_laststep = x3r[-1]
        # x3r_laststep = self.relu(x3r_laststep)
        # x3r_laststep = torch.unsqueeze(x3r_laststep, dim=0)
        # x3rplot = rearrange(x3r, 'l b c -> c b l')
        # plt.subplot(243)
        # plt.plot(x3rplot[:, 0, :].cpu().detach().numpy())

        # x4r, hids_4, cells_4 = lstm_cell_memory(x4r, 64, 64)

        h0_x4r = Variable(torch.zeros(1, x4r.size(1), self.embed_dim +25)).to(device='cuda')
        c0_x4r = Variable(torch.zeros(1, x4r.size(1), self.embed_dim +25)).to(device="cuda")
        x4r_lstm, (hn_x4r, cn_x4r) = self.lstm_2_4(x4r, (h0_x4r, c0_x4r))
        # x4r = self.gru_2_4(x4r)[0]
        # x4r_laststep = x4r[-1]
        # x4r_laststep = self.relu(x4r_laststep)
        # x4r_laststep = torch.unsqueeze(x4r_laststep, dim=0)
        # x4rplot = rearrange(x4r, 'l b c -> c b l')
        # plt.subplot(244)
        # plt.plot(x4rplot[:, 0, :].cpu().detach().numpy())

        # x5r, hids_5, cells_5 = lstm_cell_memory(x5r, 64, 64)
        h0_x5r = Variable(torch.zeros(1, x5r.size(1), self.embed_dim +25)).to(device='cuda')
        c0_x5r = Variable(torch.zeros(1, x5r.size(1), self.embed_dim +25)).to(device="cuda")
        x5r_lstm, (hn_x5r, cn_x5r) = self.lstm_2_5(x5r, (h0_x5r, c0_x5r))
        # x5r = self.gru_2_5(x5r)[0]
        # x5r_laststep = x5r[-1]
        # x5r_laststep = self.relu(x5r_laststep)
        # x5r_laststep = torch.unsqueeze(x5r_laststep, dim=0)
        # x5rplot = rearrange(x5r, 'l b c -> c b l')
        # plt.subplot(245)
        # plt.plot(x5rplot[:, 0, :].cpu().detach().numpy())

        # x6r, hids_6, cells_6 = lstm_cell_memory(x6r, 64, 64)

        h0_x6r = Variable(torch.zeros(1, x6r.size(1), self.embed_dim +25)).to(device='cuda')
        c0_x6r = Variable(torch.zeros(1, x6r.size(1), self.embed_dim +25)).to(device="cuda")
        x6r_lstm, (hn_x6r, cn_x6r) = self.lstm_2_6(x6r, (h0_x6r, c0_x6r))
        # x6r = self.gru_2_6(x6r)[0]
        # x6r_laststep = x6r[-1]
        # x6r_laststep = self.relu(x6r_laststep)
        # x6r_laststep = torch.unsqueeze(x6r_laststep, dim=0)
        # x6rplot = rearrange(x6r, 'l b c -> c b l')
        # plt.subplot(246)
        # plt.plot(x6rplot[:, 0, :].cpu().detach().numpy())

        # x7r, hids_7, cells_7 = lstm_cell_memory(x7r, 64, 64)

        h0_x7r = Variable(torch.zeros(1, x7r.size(1), self.embed_dim +25)).to(device='cuda')
        c0_x7r = Variable(torch.zeros(1, x7r.size(1), self.embed_dim +25)).to(device="cuda")
        x7r_lstm, (hn_x7r, cn_x7r) = self.lstm_2_7(x7r, (h0_x7r, c0_x7r))
        # x7r = self.gru_2_7(x7r)[0]
        # x7r_laststep = x7r[-1]
        # x7r_laststep = self.relu(x7r_laststep)
        # x7r_laststep = torch.unsqueeze(x7r_laststep, dim=0)
        # x7rplot = rearrange(x7r, 'l b c -> c b l')
        # plt.subplot(247)
        # plt.plot(x7rplot[:, 0, :].cpu().detach().numpy())

        # x8r, hids_8, cells_8 = lstm_cell_memory(x8r, 64, 64)

        h0_x8r = Variable(torch.zeros(1, x8r.size(1), self.embed_dim +25)).to(device='cuda')
        c0_x8r = Variable(torch.zeros(1, x8r.size(1), self.embed_dim +25)).to(device="cuda")
        x8r_lstm, (hn_x8r, cn_x8r) = self.lstm_2_8(x8r, (h0_x8r, c0_x8r))
        # x8r = self.gru_2_8(x8r)[0]
        # x8r_laststep = x8r[-1]
        # x8r_laststep = self.relu(x8r_laststep)
        # x8r_laststep = torch.unsqueeze(x8r_laststep, dim=0)
        # print('x8r_last',x8r_laststep.shape)
        # x8rplot = rearrange(x8r, 'l b c -> c b l')
        # plt.subplot(248)
        # plt.plot(x8rplot[:, 0, :].cpu().detach().numpy())
        # plt.show()

        # 设置transformer的参数给每个direction
        encoder_layer_1 = nn.TransformerEncoderLayer(d_model=self.embed_dim +25, nhead=1, dim_feedforward=self.embed_dim, activation='gelu').to(
            device='cuda')
        transformer_encoder_1 = nn.TransformerEncoder(encoder_layer_1, num_layers=1, norm=None).to(device='cuda')

        encoder_layer_2 = nn.TransformerEncoderLayer(d_model=self.embed_dim +25, nhead=1, dim_feedforward=self.embed_dim, activation='gelu').to(
            device='cuda')
        transformer_encoder_2 = nn.TransformerEncoder(encoder_layer_2, num_layers=1, norm=None).to(device='cuda')

        encoder_layer_3 = nn.TransformerEncoderLayer(d_model=self.embed_dim +25, nhead=1, dim_feedforward=self.embed_dim, activation='gelu').to(
            device='cuda')
        transformer_encoder_3 = nn.TransformerEncoder(encoder_layer_3, num_layers=1, norm=None).to(device='cuda')

        encoder_layer_4 = nn.TransformerEncoderLayer(d_model=self.embed_dim +25, nhead=1, dim_feedforward=self.embed_dim, activation='gelu').to(
            device='cuda')
        transformer_encoder_4 = nn.TransformerEncoder(encoder_layer_4, num_layers=1, norm=None).to(device='cuda')

        encoder_layer_5 = nn.TransformerEncoderLayer(d_model=self.embed_dim +25, nhead=1, dim_feedforward=self.embed_dim, activation='gelu').to(
            device='cuda')
        transformer_encoder_5 = nn.TransformerEncoder(encoder_layer_5, num_layers=1, norm=None).to(device='cuda')

        encoder_layer_6 = nn.TransformerEncoderLayer(d_model=self.embed_dim +25, nhead=1, dim_feedforward=self.embed_dim, activation='gelu').to(
            device='cuda')
        transformer_encoder_6 = nn.TransformerEncoder(encoder_layer_6, num_layers=1, norm=None).to(device='cuda')

        encoder_layer_7 = nn.TransformerEncoderLayer(d_model=self.embed_dim +25, nhead=1, dim_feedforward=self.embed_dim, activation='gelu').to(
            device='cuda')
        transformer_encoder_7 = nn.TransformerEncoder(encoder_layer_7, num_layers=1, norm=None).to(device='cuda')

        encoder_layer_8 = nn.TransformerEncoderLayer(d_model=self.embed_dim +25, nhead=1, dim_feedforward=self.embed_dim, activation='gelu').to(
            device='cuda')
        transformer_encoder_8 = nn.TransformerEncoder(encoder_layer_8, num_layers=1, norm=None).to(device='cuda')

        # 训练transformer
        x1r_output = transformer_encoder_1(x1r)
        var2 = torch.var(x1r_output)
        print('var2:',var2)

        '第三次TSNE'
        x1r_tsne3 = rearrange(x1r_output, 'n b c -> (n b) c').cpu().detach().numpy()
        # print('x1r_tsne1', x1r_tsne2.shape)
        x1r_tsne_out3 = TSNE(n_components=2, verbose=3).fit_transform(x1r_tsne3)

        im3 = sns.jointplot(x1r_tsne_out3[:, 0], x1r_tsne_out3[:, 1], hue=x_label, palette='Paired', s=5)
        plt.show()

        # x1r_output = self.layernorm(x1r_output)
        x1r_output = self.denselayer(x1r_output + x1r + x1r_lstm)
        # plt.subplot(122)
        # sns.heatmap(x1r_output[:, 1, :].cpu().detach().numpy())
        # plt.show()

        x2r_output = transformer_encoder_2(x2r + x2r_lstm)
        # x2r_output = self.layernorm(x2r_output)
        x2r_output = self.denselayer(x2r_output + x2r + x2r_lstm)

        x3r_output = transformer_encoder_3(x3r + x3r_lstm)
        # x3r_output = self.layernorm(x3r_output)
        x3r_output = self.denselayer(x3r_output + x3r + x3r_lstm)

        x4r_output = transformer_encoder_4(x4r + x4r_lstm)
        # x4r_output = self.layernorm(x4r_output)
        x4r_output = self.denselayer(x4r_output + x4r + x4r_lstm)

        x5r_output = transformer_encoder_5(x5r + x5r_lstm)
        # x5r_output = self.layernorm(x5r_output)
        x5r_output = self.denselayer(x5r_output + x5r + x5r_lstm)

        x6r_output = transformer_encoder_6(x6r + x6r_lstm)
        # x6r_output = self.layernorm(x6r_output)
        x6r_output = self.denselayer(x6r_output + x6r + x6r_lstm)

        x7r_output = transformer_encoder_7(x7r + x7r_lstm)
        # x7r_output = self.layernorm(x7r_output)
        x7r_output = self.denselayer(x7r_output + x7r + x7r_lstm)

        x8r_output = transformer_encoder_8(x8r + x8r_lstm)
        # x8r_output = self.layernorm(x8r_output)
        x8r_output = self.denselayer(x8r_output + x8r + x8r_lstm)
        print('1111', x1r_output.shape)  # (x,b,c) SA(25,100,self.embed_dim)


        # 提取中间像素信息
        # x1r_output_central = x1r_output[0, :, :] + x1r_output[13,:,:]
        # x1r_output_clstoken = x1r_output[0, :, :]
        x1r_output_centraltoken = (x1r_output + x1r_lstm)[13, :, :]
        x1r_output_meantoken = reduce(x1r_output, 'x b c -> b c', reduction='mean')
        # print('1111',x1r_output_central.shape) #(b,c) SA(100, 204)\

        # x2r_output_central = x2r_output[0, :, :] + x2r_output[13,:,:]
        # x2r_output_clstoken = x2r_output[0, :, :]
        x2r_output_centraltoken = (x2r_output + x2r_lstm)[13, :, :]
        x2r_output_meantoken = reduce(x2r_output, 'x b c -> b c', reduction='mean')

        # x3r_output_central = x3r_output[0, :, :] + x3r_output[13,:,:]
        # x3r_output_clstoken = x3r_output[0, :, :]
        x3r_output_centraltoken = (x3r_output+x3r_lstm)[13, :, :]
        x3r_output_meantoken = reduce(x3r_output, 'x b c -> b c', reduction='mean')

        # x4r_output_central = x4r_output[0, :, :] + x4r_output[13,:,:]
        # x4r_output_clstoken = x4r_output[0, :, :]
        x4r_output_centraltoken = (x4r_output+x4r_lstm)[13, :, :]
        x4r_output_meantoken = reduce(x4r_output, 'x b c -> b c', reduction='mean')

        # x5r_output_central = x5r_output[0, :, :] + x5r_output[13,:,:]
        # x5r_output_clstoken = x5r_output[0, :, :]
        x5r_output_centraltoken = (x5r_output + x5r_lstm)[13, :, :]
        x5r_output_meantoken = reduce(x5r_output, 'x b c -> b c', reduction='mean')

        # x6r_output_central = x6r_output[0, :, :] + x6r_output[13,:,:]
        # x6r_output_clstoken = x6r_output[0, :, :]
        x6r_output_centraltoken = (x6r_output + x6r_lstm)[13, :, :]
        x6r_output_meantoken = reduce(x6r_output, 'x b c -> b c', reduction='mean')

        # x7r_output_central = x7r_output[0, :, :] + x7r_output[13,:,:]
        # x7r_output_clstoken = x7r_output[0, :, :]
        x7r_output_centraltoken = (x7r_output + x7r_lstm)[13, :, :]
        x7r_output_meantoken = reduce(x7r_output, 'x b c -> b c', reduction='mean')

        # x8r_output_central = x8r_output[0, :, :] + x8r_output[13,:,:]
        # x8r_output_clstoken = x8r_output[0, :, :]
        x8r_output_centraltoken = (x8r_output+ x8r_lstm)[13, :, :]
        x8r_output_meantoken = reduce(x8r_output, 'x b c -> b c', reduction='mean')

        # 扩展维度准备合并
        # x1r_output_centraltoken = rearrange(x1r_output_centraltoken, 'b c -> () b c')
        # x2r_output_centraltoken = rearrange(x2r_output_centraltoken, 'b c -> () b c')
        # x3r_output_centraltoken = rearrange(x3r_output_centraltoken, 'b c -> () b c')
        # x4r_output_centraltoken = rearrange(x4r_output_centraltoken, 'b c -> () b c')
        # x5r_output_centraltoken = rearrange(x5r_output_centraltoken, 'b c -> () b c')
        # x6r_output_centraltoken = rearrange(x6r_output_centraltoken, 'b c -> () b c')
        # x7r_output_centraltoken = rearrange(x7r_output_centraltoken, 'b c -> () b c')
        # x8r_output_centraltoken = rearrange(x8r_output_centraltoken, 'b c -> () b c')
        # print('x1r_output_centraltoken', x1r_output_centraltoken.shape)
        #
        # x1r_output_meantoken = rearrange(x1r_output_meantoken, 'b c -> () b c')
        # x2r_output_meantoken = rearrange(x2r_output_meantoken, 'b c -> () b c')
        # x3r_output_meantoken = rearrange(x3r_output_meantoken, 'b c -> () b c')
        # x4r_output_meantoken = rearrange(x4r_output_meantoken, 'b c -> () b c')
        # x5r_output_meantoken = rearrange(x5r_output_meantoken, 'b c -> () b c')
        # x6r_output_meantoken = rearrange(x6r_output_meantoken, 'b c -> () b c')
        # x7r_output_meantoken = rearrange(x7r_output_meantoken, 'b c -> () b c')
        # x8r_output_meantoken = rearrange(x8r_output_meantoken, 'b c -> () b c')
        # print('x1r_output_meantoken', x1r_output_meantoken.shape)

        '只用一个扫描 or 2 or 4 '
        # x1r_output_conca = torch.cat([x1r_output_meantoken, x1r_output_centraltoken], dim=1)
        # x7r_output_conca = torch.cat([x7r_output_meantoken, x7r_output_centraltoken], dim=1)
        # x2r_output_conca = torch.cat([x2r_output_meantoken, x2r_output_centraltoken], dim=1)
        # x8r_output_conca = torch.cat([x8r_output_meantoken, x8r_output_centraltoken], dim=1)
        # x3r_output_conca = torch.cat([x3r_output_meantoken, x3r_output_centraltoken], dim=1)
        # x4r_output_conca = torch.cat([x4r_output_meantoken, x4r_output_centraltoken], dim=1)
        # x5r_output_conca = torch.cat([x5r_output_meantoken, x5r_output_centraltoken], dim=1)
        # x6r_output_conca = torch.cat([x6r_output_meantoken, x6r_output_centraltoken], dim=1)

        x1r_output_conca = repeat(x1r_output_centraltoken, 'b c -> b c ()')
        x7r_output_conca = repeat(x7r_output_centraltoken, 'b c -> b c ()')
        x2r_output_conca = repeat(x2r_output_centraltoken, 'b c -> b c ()')
        x8r_output_conca = repeat(x8r_output_centraltoken, 'b c -> b c ()')
        x3r_output_conca = repeat(x3r_output_centraltoken, 'b c -> b c ()')
        x4r_output_conca = repeat(x4r_output_centraltoken, 'b c -> b c ()')
        x5r_output_conca = repeat(x5r_output_centraltoken, 'b c -> b c ()')
        x6r_output_conca = repeat(x6r_output_centraltoken, 'b c -> b c ()')

        preds_onedirection = torch.cat(
            [x1r_output_conca, x7r_output_conca, x2r_output_conca, x8r_output_conca, x3r_output_conca, x5r_output_conca,
             x4r_output_conca, x6r_output_conca], dim=2)  # （b c x)

        # # x1r = self.gru_2_1(x1r)[0]
        # print('x1r', x1r.shape)  #(x,b,c) (25,100,64)
        # x1r_laststep = x1r[-1]
        # x1r_laststep = self.relu(x1r_laststep)
        # x1r_laststep = torch.unsqueeze(x1r_laststep, dim=0)

        # print('```',preds_onedirection.shape)
        #

        preds_onedirection = preds_onedirection.view(preds_onedirection.size(0), -1)
        print('final1', preds_onedirection.shape)
        preds_onedirection = self.transformer_bn_scheme1(preds_onedirection)
        preds_onedirection = self.relu(preds_onedirection)
        preds_onedirection = self.dropout(preds_onedirection)
        preds_onedirection = self.fc_transformer_scheme1(preds_onedirection)
        print('final2',preds_onedirection.shape)


        "用LSTM"
        # h0_x1r = torch.zeros(1, x1r.size(1), 64).to(device='cuda')
        # c0_x1r = torch.zeros(1, x1r.size(1), 64).to(device="cuda")
        # x1r, (hn_x1r, cn_x1r) = self.lstm_2_1(x1r, (h0_x1r, c0_x1r))
        # # x1r = self.gru_2_1(x1r)[0]
        # print('x1r', x1r.shape)  #(x,b,c) (25,100,64)
        # x1r_laststep = x1r[-1]
        # x1r_laststep = self.relu(x1r_laststep)
        # x1r_laststep = torch.unsqueeze(x1r_laststep, dim=0)
        #
        # h0_x2r = torch.zeros(1, x2r.size(1), 64).to(device='cuda')
        # c0_x2r = torch.zeros(1, x2r.size(1), 64).to(device="cuda")
        # x2r, (hn_x2r, cn_x2r) = self.lstm_2_2(x2r, (h0_x2r, c0_x2r))
        # # x2r = self.gru_2_2(x2r)[0] #把x1r经过RNN的值，作为x2r的输入
        # x2r_laststep = x2r[-1]
        # x2r_laststep = self.relu(x2r_laststep)
        # x2r_laststep = torch.unsqueeze(x2r_laststep, dim=0)
        #
        # h0_x3r = torch.zeros(1, x3r.size(1), 64).to(device='cuda')
        # c0_x3r = torch.zeros(1, x3r.size(1), 64).to(device="cuda")
        # x3r, (hn_x3r, cn_x3r) = self.lstm_2_3(x3r, (h0_x3r, c0_x3r))
        # # x3r = self.gru_2_3(x3r)[0]
        # x3r_laststep = x3r[-1]
        # x3r_laststep = self.relu(x3r_laststep)
        # x3r_laststep = torch.unsqueeze(x3r_laststep, dim=0)
        #
        # h0_x4r = torch.zeros(1, x4r.size(1), 64).to(device='cuda')
        # c0_x4r = torch.zeros(1, x4r.size(1), 64).to(device="cuda")
        # x4r, (hn_x4r, cn_x4r) = self.lstm_2_4(x4r, (h0_x4r, c0_x4r))
        # # x4r = self.gru_2_4(x4r)[0]
        # x4r_laststep = x4r[-1]
        # x4r_laststep = self.relu(x4r_laststep)
        # x4r_laststep = torch.unsqueeze(x4r_laststep, dim=0)
        #
        # h0_x5r = torch.zeros(1, x5r.size(1), 64).to(device='cuda')
        # c0_x5r = torch.zeros(1, x5r.size(1), 64).to(device="cuda")
        # x5r, (hn_x5r, cn_x5r) = self.lstm_2_5(x5r, (h0_x5r, c0_x5r))
        # # x5r = self.gru_2_5(x5r)[0]
        # x5r_laststep = x5r[-1]
        # x5r_laststep = self.relu(x5r_laststep)
        # x5r_laststep = torch.unsqueeze(x5r_laststep, dim=0)
        #
        # h0_x6r = torch.zeros(1, x6r.size(1), 64).to(device='cuda')
        # c0_x6r = torch.zeros(1, x6r.size(1), 64).to(device="cuda")
        # x6r, (hn_x6r, cn_x6r) = self.lstm_2_6(x6r, (h0_x6r, c0_x6r))
        # # x6r = self.gru_2_6(x6r)[0]
        # x6r_laststep = x6r[-1]
        # x6r_laststep = self.relu(x6r_laststep)
        # x6r_laststep = torch.unsqueeze(x6r_laststep, dim=0)
        #
        # h0_x7r = torch.zeros(1, x7r.size(1), 64).to(device='cuda')
        # c0_x7r = torch.zeros(1, x7r.size(1), 64).to(device="cuda")
        # x7r, (hn_x7r, cn_x7r) = self.lstm_2_7(x7r, (h0_x7r, c0_x7r))
        # # x7r = self.gru_2_7(x7r)[0]
        # x7r_laststep = x7r[-1]
        # x7r_laststep = self.relu(x7r_laststep)
        # x7r_laststep = torch.unsqueeze(x7r_laststep, dim=0)
        #
        # h0_x8r = torch.zeros(1, x8r.size(1), 64).to(device='cuda')
        # c0_x8r = torch.zeros(1, x8r.size(1), 64).to(device="cuda")
        # x8r, (hn_x8r, cn_x8r) = self.lstm_2_8(x8r, (h0_x8r, c0_x8r))
        # # x8r = self.gru_2_8(x8r)[0]
        # x8r_laststep = x8r[-1]
        # x8r_laststep = self.relu(x8r_laststep)
        # x8r_laststep = torch.unsqueeze(x8r_laststep, dim=0)
        # print('x8r_last',x8r_laststep.shape)

        "scheme 1"
        # x_strategy_FLC = torch.cat([x1r_output_central,x2r_output_central,x3r_output_central,x4r_output_central,x5r_output_central,x6r_output_central,x7r_output_central,x8r_output_central],dim=0)
        # print('x_strategy_FLC', x_strategy_FLC.shape) # (x, b, c) (8 , batch, 64)
        # x_strategy_FLC = rearrange(x_strategy_FLC, 'x b c -> b x c')
        # x_strategy_FLC += self.pos_embedding_conca_scheme1[:, :(x_strategy_FLC.size(1))]
        # x_strategy_FLC = rearrange(x_strategy_FLC, 'b x c -> x b c')
        # # 设置transformer的参数给8个direction
        # encoder_layer_conca = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=1, dim_feedforward=32, activation='gelu').to(
        #     device='cuda')
        # transformer_encoder_conca = nn.TransformerEncoder(encoder_layer_conca, num_layers=1, norm=None).to(device='cuda')
        #
        # x_strategy_FLC_output = transformer_encoder_conca(x_strategy_FLC)
        # x_strategy_FLC_output = self.denselayer_scheme1(x_strategy_FLC_output)
        # x_strategy_FLC_output = reduce(x_strategy_FLC_output, 'x b c -> 1 b c', reduction='mean')
        # x_strategy_FLC_output = self.relu(x_strategy_FLC_output)
        # x_strategy_FLC_output = x_strategy_FLC_output.view(x_strategy_FLC_output.size(1), -1)
        # x_strategy_FLC_output = self.transformer_bn_scheme1(x_strategy_FLC_output)
        # x_strategy_FLC_output = self.dropout(x_strategy_FLC_output)
        # preds_scheme1 = self.fc_transformer_scheme1(x_strategy_FLC_output)

        "scheme 2"
        # x1_7r = torch.cat([x1r_output_conca, x7r_output_conca], dim=1 )
        # x2_8r = torch.cat([x2r_output_conca, x8r_output_conca], dim=1 )
        # x3_5r = torch.cat([x3r_output_conca, x5r_output_conca], dim=1 )
        # x4_6r = torch.cat([x4r_output_conca, x6r_output_conca], dim=1 )
        # x_strategy_FLC = torch.cat([x1_7r, x2_8r, x3_5r, x4_6r], dim=2)
        # x_strategy_FLC = rearrange(x_strategy_FLC, 'b c x -> b x c')
        # print('...', x_strategy_FLC.shape)
        # cls_tokens_FLC = repeat(self.cls_token_FLC, '() n c -> b n c', b=b)
        # x_strategy_FLC = torch.cat((cls_tokens_FLC, x_strategy_FLC), dim=1)
        # x_strategy_FLC += self.pos_embedding_conca_scheme2[:, :(x_strategy_FLC.size(1) + 1 )]
        # x_strategy_FLC = rearrange(x_strategy_FLC, 'b x c -> x b c')
        # # 设置transformer的参数给4对directions
        # encoder_layer_conca = nn.TransformerEncoderLayer(d_model=self.embed_dim * 4, nhead=1, dim_feedforward=64,
        #                                                  activation='gelu').to(device='cuda')
        # transformer_encoder_conca = nn.TransformerEncoder(encoder_layer_conca, num_layers=1, norm=None).to(device='cuda')
        # x_strategy_FLC_output = transformer_encoder_conca(x_strategy_FLC)
        # x_strategy_FLC_output = self.denselayer_scheme2(x_strategy_FLC_output)
        #
        # x_strategy_FLC_output_clstoken = x_strategy_FLC_output[0,:,:]
        # x_strategy_FLC_output_meantoken = (x_strategy_FLC_output[1,:,:] + x_strategy_FLC_output[2,:,:] + x_strategy_FLC_output[3,:,:] + x_strategy_FLC_output[4,:,:])
        # # x_strategy_FLC_output = reduce(x_strategy_FLC_output, 'x b c -> 1 b c', reduction='mean')
        #
        # x_strategy_FLC_output = self.relu(x_strategy_FLC_output_meantoken)
        # x_strategy_FLC_output = x_strategy_FLC_output.view(x_strategy_FLC_output.size(0), -1)
        # x_strategy_FLC_output = self.transformer_bn_scheme2(x_strategy_FLC_output)
        # x_strategy_FLC_output = self.dropout(x_strategy_FLC_output)
        # preds_scheme2 = self.fc_transformer_scheme2(x_strategy_FLC_output)

        return preds_onedirection


class zhou_single_multi_scanning_Trans(nn.Module):

    @staticmethod
    def weight_init(m):
        # All weight matrices in our RNN and bias vectors are initialized with a uniform distribution, and the values of these weight matrices and bias vectors are initialized in the range [−0.1,0.1]
        if isinstance(m, (nn.Linear, nn.GRU, nn.Conv3d,nn.LSTM, nn.TransformerEncoderLayer, nn.TransformerEncoder)):
            init.kaiming_uniform_(m.weight.data)
            init.kaiming_uniform_(m.bias.data)

    def __init__(self, input_channels, n_classes, patch_size=5, embed_dim = 64):
        super(zhou_single_multi_scanning_Trans, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        'do not use self.n_classes = n_classes'
        self.lstm_2_1 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.lstm_2_2 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.lstm_2_3 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.lstm_2_4 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.lstm_2_5 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.lstm_2_6 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.lstm_2_7 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.lstm_2_8 = nn.LSTM(input_channels, 64, 1, bidirectional=False)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding_1 = nn.Parameter(torch.randn(1, 25 + 1, embed_dim))
        self.pos_embedding_2 = nn.Parameter(torch.randn(1, 25 +1, embed_dim))
        self.pos_embedding_3 = nn.Parameter(torch.randn(1, 25 +1, embed_dim))
        self.pos_embedding_4 = nn.Parameter(torch.randn(1, 25 +1, embed_dim))
        self.pos_embedding_5 = nn.Parameter(torch.randn(1, 25 +1, embed_dim))
        self.pos_embedding_6 = nn.Parameter(torch.randn(1, 25 +1, embed_dim))
        self.pos_embedding_7 = nn.Parameter(torch.randn(1, 25 +1, embed_dim))
        self.pos_embedding_8 = nn.Parameter(torch.randn(1, 25 +1, embed_dim))
        self.pos_embedding_conca_scheme1 = nn.Parameter(torch.randn(1, 8, embed_dim))
        self.pos_embedding_conca_scheme2 = nn.Parameter(torch.randn(1, 4 + 1, embed_dim * 4))
        self.cls_token_FLC = nn.Parameter(torch.randn(1, 1, embed_dim * 4))
        self.lstm_4 = nn.LSTM(patch_size ** 2, 64, 1)
        self.transformer = nn.Transformer(nhead=16, num_encoder_layers=12)
        self.conv2d_1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=2, stride=1)
        self.conv2d_2 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=2, stride=1)
        self.conv2d_3 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, stride=1)
        self.conv2d_4 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1)
        self.conv7to5 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, stride=1)
        self.conv7to5_2 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, stride=1)
        self.deconv5to15 = nn.ConvTranspose2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, stride=3)
        self.gru_bn = nn.BatchNorm1d(patch_size ** 2 * input_channels)
        self.lstm_bn_1 = nn.BatchNorm1d((64)*1)
        self.lstm_bn_1_2 = nn.BatchNorm1d((64) * (patch_size)**2)
        self.lstm_bn_2 = nn.BatchNorm1d((64)*8)
        self.lstm_bn_2_2 = nn.BatchNorm1d((64) * 8 * patch_size**2)
        self.gru_bn_2 = nn.BatchNorm1d((patch_size ** 2) * (patch_size**2))
        self.gru_bn_3 = nn.BatchNorm1d((patch_size ** 2) * (patch_size ** 2) * 2)
        self.gru_bn_4 = nn.BatchNorm1d(8 * 64)
        self.gru_bn_laststep = nn.BatchNorm1d(64)
        self.transformer_bn_scheme1 = nn.BatchNorm1d(embed_dim * 16 )
        self.transformer_bn_scheme2 = nn.BatchNorm1d(embed_dim * 2)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(patch_size ** 2 * input_channels, n_classes)
        self.lstm_fc_1 = nn.Linear(64 * 1, n_classes)
        self.lstm_fc_1_2 = nn.Linear(64 * (patch_size**2), n_classes)
        self.lstm_fc_2 = nn.Linear(64*8,n_classes)
        self.lstm_fc_2_2 = nn.Linear(64 * 8 * patch_size**2, n_classes)
        self.fc_2 = nn.Linear((patch_size ** 2) * (patch_size**2), n_classes)
        self.fc_3 = nn.Linear((patch_size ** 2) * (patch_size ** 2) * 2, n_classes)
        self.fc_4 = nn.Linear(8 * 64, n_classes)
        self.layernorm = nn.LayerNorm(embed_dim)
        self.denselayer = nn.Linear(embed_dim,embed_dim)
        self.denselayer_scheme1 = nn.Linear(embed_dim,embed_dim)
        self.denselayer_scheme2 = nn.Linear(embed_dim * 4, embed_dim * 2)
        self.fc_laststep = nn.Linear(64, n_classes)
        self.fc_transformer_scheme1 = nn.Linear(embed_dim *16,n_classes)
        self.fc_transformer_scheme2 = nn.Linear(embed_dim * 2 , n_classes)
        self.linearembedding = nn.Linear(in_features=input_channels,out_features=embed_dim)
        self.softmax = nn.Softmax()
        self.localnorm = nn.LocalResponseNorm(input_channels)

    def forward(self, x): #初始是第1方向
        print('x.shape1',x.shape)
        x = x.squeeze(1) #b,c,h w
        # x_7to5 = self.conv7to5(x)
        # x = x_7to5

        # x_show1 = self.relu(self.conv7to5_2(x_7to5))
        # x_show2 = self.relu(self.deconv5to15(x))
        # print('x_show',x_show1.shape)
        # plt.subplot(121)
        # plt.imshow(x_show1[0, 1, :, :].cpu().detach().numpy())
        # plt.subplot(122)
        # plt.imshow(x_show2[0,1,:,:].cpu().detach().numpy())
        # plt.show()


        # ResNet patch_size = 9 for SA PU
        # x = self.conv2d_1(x)
        # print('1', x.shape)
        # x = self.relu(x)
        # x = self.conv2d_2(x)
        # print('2', x.shape)
        # x_res = self.relu(x)
        # x_res = self.conv2d_3(x_res)
        # print('3', x.shape) #(ptach size = 6)
        # x_res = self.relu(x_res)
        # x_res_res = self.conv2d_4(x_res)
        # x_res_res = self.relu(x_res_res)
        # x = x_res + x_res_res
        # print('4', x.shape)

        #生成第1和7
        x1_0 = x[:, :, 0, :]
        x1_1 = x[:, :, 1, :]
        x1_2 = x[:, :, 2, :]
        x1_3 = x[:, :, 3, :]
        x1_4 = x[:, :, 4, :]
        x1_1f = torch.flip(x1_1, [2])
        x1_3f = torch.flip(x1_3, [2])
        # direction_1 = torch.cat([x1_0, x1_1f, x1_2], dim=2)
        direction_1 = torch.cat([x1_0, x1_1f, x1_2,x1_3f,x1_4], dim=2)

        print('d1',direction_1.shape)
        # print('d1',direction_1.shape)
        direction_7 = torch.flip(direction_1,[2])

        #生成第2和8
        x2_0 = x[:, :, :, 0]
        x2_1 = x[:, :, :, 1]
        x2_2 = x[:, :, :, 2]
        x2_3 = x[:, :, :, 3]
        x2_4 = x[:, :, :, 4]
        x2_1f = torch.flip(x2_1, [2])
        x2_3f = torch.flip(x2_3, [2])
        # direction_2 = torch.cat([x2_0, x2_1f, x2_2], dim=2)
        direction_2 = torch.cat([x2_0, x2_1f,x2_2,x2_3f,x2_4], dim=2)
        direction_8 = torch.flip(direction_2, [2])

        # 生成3和5
        x3_0 = x[:, :, 0, :]
        x3_1 = x[:, :, 1, :]
        x3_2 = x[:, :, 2, :]
        x3_3 = x[:, :, 3, :]
        x3_4 = x[:, :, 4, :]
        x3_0f = torch.flip(x3_0, [2])
        x3_2f = torch.flip(x3_2, [2])
        x3_4f = torch.flip(x3_4, [2])
        # direction_3 = torch.cat([x3_0f, x3_1, x3_2f], dim=2)
        direction_3 = torch.cat([x3_0f, x3_1, x3_2f,x3_3,x3_4f], dim=2)
        direction_5 = torch.flip(direction_3, [2])

        #生成4和6
        x4_0 = x[:, :, :, 0]
        x4_1 = x[:, :, :, 1]
        x4_2 = x[:, :, :, 2]
        x4_3 = x[:, :, :, 3]
        x4_4 = x[:, :, :, 4]
        x4_1f = torch.flip(x4_1, [2])
        x4_3f = torch.flip(x4_3, [2])
        # direction_4 = torch.cat([x4_2, x4_1f, x4_0], dim=2)
        direction_4 = torch.cat([x4_4, x4_3f, x4_2, x4_1f, x4_0], dim=2)
        # print('d4', direction_4.shape)
        direction_6 = torch.flip(direction_4, [2])

        # plt.xlabel('Band Numbers', fontdict={'size': 35}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 35}, fontweight='bold')
        # plt.plot(direction_1[0, :, 0].cpu().detach().numpy(), label='index 0')
        # plt.plot(direction_1[0, :, 1].cpu().detach().numpy(), label='index 1')
        # plt.plot(direction_1[0, :, 2].cpu().detach().numpy(), label='index 2')
        # plt.plot(direction_1[0, :, 3].cpu().detach().numpy(), label='index 3')
        # plt.plot(direction_1[0, :, 4].cpu().detach().numpy(), label='index 4')
        # plt.plot(direction_1[0, :, 5].cpu().detach().numpy(), label='index 9')
        # plt.plot(direction_1[0, :, 6].cpu().detach().numpy(), label='index 8')
        # plt.plot(direction_1[0, :, 7].cpu().detach().numpy(), label='index 7')
        # plt.plot(direction_1[0, :, 8].cpu().detach().numpy(), label='index 6')
        # plt.plot(direction_1[0, :, 9].cpu().detach().numpy(), label='index 5')
        # plt.plot(direction_1[0, :, 10].cpu().detach().numpy(), label='index 10')
        # plt.plot(direction_1[0, :, 11].cpu().detach().numpy(), label='index 11')
        # plt.plot(direction_1[0, :, 12].cpu().detach().numpy(), label='index 12', linewidth=5, linestyle='-.', color = 'red' )
        # plt.plot(direction_1[0, :, 13].cpu().detach().numpy(), label='index 13')
        # plt.plot(direction_1[0, :, 14].cpu().detach().numpy(), label='index 14')
        # plt.plot(direction_1[0, :, 15].cpu().detach().numpy(), label='index 19')
        # plt.plot(direction_1[0, :, 16].cpu().detach().numpy(), label='index 18')
        # plt.plot(direction_1[0, :, 17].cpu().detach().numpy(), label='index 17')
        # plt.plot(direction_1[0, :, 18].cpu().detach().numpy(), label='index 16')
        # plt.plot(direction_1[0, :, 19].cpu().detach().numpy(), label='index 15')
        # plt.plot(direction_1[0, :, 20].cpu().detach().numpy(), label='index 20')
        # plt.plot(direction_1[0, :, 21].cpu().detach().numpy(), label='index 21')
        # plt.plot(direction_1[0, :, 22].cpu().detach().numpy(), label='index 22')
        # plt.plot(direction_1[0, :, 23].cpu().detach().numpy(), label='index 23')
        # plt.plot(direction_1[0, :, 24].cpu().detach().numpy(), label='index 24')
        # plt.legend(prop={'size': 15}, fontsize='large', loc=[1, 0])
        # plt.title('Spectral signatures in a patch', fontdict={'size': 35}, fontweight='bold')
        # plt.grid(linewidth = 1.5)
        # plt.xticks(fontsize=25)
        # plt.yticks(fontsize=25)
        # plt.show()
        # plt.subplot(122)
        # plt.xlabel('Band Numbers', fontdict={'size': 35}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 35}, fontweight='bold')
        # plt.plot(direction_2[0, :, 0].cpu().detach().numpy(), label='(0,0),0')
        # plt.plot(direction_2[0, :, 1].cpu().detach().numpy(), label='(1,0),5')
        # plt.plot(direction_2[0, :, 2].cpu().detach().numpy(), label='(2,0),10')
        # plt.plot(direction_2[0, :, 3].cpu().detach().numpy(), label='(3,0),15')
        # plt.plot(direction_2[0, :, 4].cpu().detach().numpy(), label='(4,0),20')
        # plt.plot(direction_2[0, :, 5].cpu().detach().numpy(), label='(4,1),21')
        # plt.plot(direction_2[0, :, 6].cpu().detach().numpy(), label='(3,1),16')
        # plt.plot(direction_2[0, :, 7].cpu().detach().numpy(), label='(2,1),11')
        # plt.plot(direction_2[0, :, 8].cpu().detach().numpy(), label='(1,1),6')
        # plt.plot(direction_2[0, :, 9].cpu().detach().numpy(), label='(0,1),1')
        # plt.plot(direction_2[0, :, 10].cpu().detach().numpy(), label='(0,2),2')
        # plt.plot(direction_2[0, :, 11].cpu().detach().numpy(), label='(1,2),7')
        # plt.plot(direction_2[0, :, 12].cpu().detach().numpy(), label='(2,2), center', linewidth=3, linestyle='-.')
        # plt.plot(direction_2[0, :, 13].cpu().detach().numpy(), label='(3,2),17')
        # plt.plot(direction_2[0, :, 14].cpu().detach().numpy(), label='(4,2),22')
        # plt.plot(direction_2[0, :, 15].cpu().detach().numpy(), label='(4,3),23')
        # plt.plot(direction_2[0, :, 16].cpu().detach().numpy(), label='(3,3),18')
        # plt.plot(direction_2[0, :, 17].cpu().detach().numpy(), label='(2,3),13')
        # plt.plot(direction_2[0, :, 18].cpu().detach().numpy(), label='(1,3),8')
        # plt.plot(direction_2[0, :, 19].cpu().detach().numpy(), label='(0,3),3', linewidth=5)
        # plt.plot(direction_2[0, :, 20].cpu().detach().numpy(), label='(0,4),4', linewidth=5)
        # plt.plot(direction_2[0, :, 21].cpu().detach().numpy(), label='(1,4),9', linewidth=5)
        # plt.plot(direction_2[0, :, 22].cpu().detach().numpy(), label='(2,4),14')
        # plt.plot(direction_2[0, :, 23].cpu().detach().numpy(), label='(3,4),19')
        # plt.plot(direction_2[0, :, 24].cpu().detach().numpy(), label='(4,4),24')
        # plt.legend(prop={'size': 15}, fontsize='large', loc=[1, 0])
        # plt.title('Spectral signatures in a patch', fontdict={'size': 35}, fontweight='bold')
        # plt.xticks(fontsize=25)
        # plt.yticks(fontsize=25)
        # plt.show()

        # # plt.subplot(332)
        # plt.imshow(direction_1[0, :, (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-1 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(333)
        # plt.imshow(direction_2[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-2 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(334)
        # plt.imshow(direction_3[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-3 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(335)
        # plt.imshow(direction_4[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-4 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(336)
        # plt.imshow(direction_5[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-5 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(337)
        # plt.imshow(direction_6[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-6 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(338)
        # plt.imshow(direction_7[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()ticks(fontsize=20)
        # plt.title('Direction-7 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()
        #
        # # plt.subplot(339)
        # plt.imshow(direction_8[0, :,
        #            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)].cpu(), cmap='gray')
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.title('Direction-8 scanning', fontdict={'size': 20}, fontweight='bold')
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # # plt.show()

        #换成输入顺序
        x8r = direction_8.permute(2, 0, 1)
        x7r = direction_7.permute(2, 0, 1)
        x6r = direction_6.permute(2, 0, 1)
        x5r = direction_5.permute(2, 0, 1)
        x4r = direction_4.permute(2, 0, 1)
        x3r = direction_3.permute(2, 0, 1)
        x2r = direction_2.permute(2, 0, 1)
        x1r = direction_1.permute(2, 0, 1)
        print('d5.shape', x5r.shape)

        print('x1r',x1r.shape) #(x,b,c) (25,100,204)

        #linear embedding
        x1r = self.linearembedding(x1r)
        x2r = self.linearembedding(x2r)
        x3r = self.linearembedding(x3r)
        x4r = self.linearembedding(x4r)
        x5r = self.linearembedding(x5r)
        x6r = self.linearembedding(x6r)
        x7r = self.linearembedding(x7r)
        x8r = self.linearembedding(x8r)

        #positional embedding
        x1r = rearrange(x1r, 'x b c -> b x c')
        b, n, c = x1r.shape
        cls_tokens_x1r = repeat(self.cls_token,  '() n c -> b n c', b = b)
        x1r = torch.cat((cls_tokens_x1r, x1r), dim=1)
        print('cls token', x1r.shape)  # (b,x,c) SA(100, 25, 204)
        x1r += self.pos_embedding_1[:, :(x1r.size(1) + 1 )]
        print('111',x1r.shape) #(b,x,c) SA(100, 25, 204)

        x2r = rearrange(x2r, 'x b c -> b x c')
        cls_tokens_x2r = repeat(self.cls_token,  '() n c -> b n c', b = b)
        x2r = torch.cat((cls_tokens_x2r, x2r), dim=1)
        x2r += self.pos_embedding_2[:, :(x2r.size(1) +1)]

        x3r = rearrange(x3r, 'x b c -> b x c')
        cls_tokens_x3r = repeat(self.cls_token,  '() n c -> b n c', b = b)
        x3r = torch.cat((cls_tokens_x3r, x3r), dim=1)
        x3r += self.pos_embedding_3[:, :(x3r.size(1)+1)]

        x4r = rearrange(x4r, 'x b c -> b x c')
        cls_tokens_x4r = repeat(self.cls_token,  '() n c -> b n c', b = b)
        x4r = torch.cat((cls_tokens_x4r, x4r), dim=1)
        x4r += self.pos_embedding_4[:, :(x4r.size(1)+1)]

        x5r = rearrange(x5r, 'x b c -> b x c')
        cls_tokens_x5r = repeat(self.cls_token,  '() n c -> b n c', b = b)
        x5r = torch.cat((cls_tokens_x5r, x5r), dim=1)
        x5r += self.pos_embedding_5[:, :(x5r.size(1)+1)]

        x6r = rearrange(x6r, 'x b c -> b x c')
        cls_tokens_x6r = repeat(self.cls_token,  '() n c -> b n c', b = b)
        x6r = torch.cat((cls_tokens_x6r, x6r), dim=1)
        x6r += self.pos_embedding_6[:, :(x6r.size(1)+1)]

        x7r = rearrange(x7r, 'x b c -> b x c')
        cls_tokens_x7r = repeat(self.cls_token,  '() n c -> b n c', b = b)
        x7r = torch.cat((cls_tokens_x7r, x7r), dim=1)
        x7r += self.pos_embedding_7[:, :(x7r.size(1)+1)]

        x8r = rearrange(x8r, 'x b c -> b x c')
        cls_tokens_x8r = repeat(self.cls_token,  '() n c -> b n c', b = b)
        x8r = torch.cat((cls_tokens_x8r, x8r), dim=1)
        x8r += self.pos_embedding_8[:, :(x8r.size(1)+1)]

        # plt.subplot(241)
        # plt.imshow(self.pos_embedding_1[0, :, :].cpu().detach().numpy())
        #
        #
        # plt.subplot(242)
        # plt.imshow(self.pos_embedding_2[0, :, :].cpu().detach().numpy())
        #
        # plt.subplot(243)
        # plt.imshow(self.pos_embedding_3[0, :, :].cpu().detach().numpy())
        #
        # plt.subplot(244)
        # plt.imshow(self.pos_embedding_4[0, :, :].cpu().detach().numpy())
        #
        # plt.subplot(245)
        # plt.imshow(self.pos_embedding_5[0, :, :].cpu().detach().numpy())
        #
        # plt.subplot(246)
        # plt.imshow(self.pos_embedding_6[0, :, :].cpu().detach().numpy())
        #
        # plt.subplot(247)
        # plt.imshow(self.pos_embedding_7[0, :, :].cpu().detach().numpy())
        #
        # plt.subplot(248)
        # plt.imshow(self.pos_embedding_8[0, :, :].cpu().detach().numpy())

        # plt.show()


        x1r = rearrange(x1r, 'b x c -> x b c')
        x2r = rearrange(x2r, 'b x c -> x b c')
        x3r = rearrange(x3r, 'b x c -> x b c')
        x4r = rearrange(x4r, 'b x c -> x b c')
        x5r = rearrange(x5r, 'b x c -> x b c')
        x6r = rearrange(x6r, 'b x c -> x b c')
        x7r = rearrange(x7r, 'b x c -> x b c')
        x8r = rearrange(x8r, 'b x c -> x b c')

        #设置transformer的参数给每个direction
        encoder_layer_1 = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=1, dim_feedforward=64,activation='gelu').to(device='cuda')
        transformer_encoder_1 = nn.TransformerEncoder(encoder_layer_1, num_layers=1, norm=None).to(device='cuda')
        encoder_layer_2 = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=1, dim_feedforward=64, activation='gelu').to(
            device='cuda')
        transformer_encoder_2 = nn.TransformerEncoder(encoder_layer_2, num_layers=1, norm=None).to(device='cuda')
        encoder_layer_3 = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=1, dim_feedforward=64, activation='gelu').to(
            device='cuda')
        transformer_encoder_3 = nn.TransformerEncoder(encoder_layer_3, num_layers=1, norm=None).to(device='cuda')
        encoder_layer_4 = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=1, dim_feedforward=64, activation='gelu').to(
            device='cuda')
        transformer_encoder_4 = nn.TransformerEncoder(encoder_layer_4, num_layers=1, norm=None).to(device='cuda')
        encoder_layer_5 = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=1, dim_feedforward=64, activation='gelu').to(
            device='cuda')
        transformer_encoder_5 = nn.TransformerEncoder(encoder_layer_5, num_layers=1, norm=None).to(device='cuda')
        encoder_layer_6 = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=1, dim_feedforward=64, activation='gelu').to(
            device='cuda')
        transformer_encoder_6 = nn.TransformerEncoder(encoder_layer_6, num_layers=1, norm=None).to(device='cuda')
        encoder_layer_7 = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=1, dim_feedforward=64, activation='gelu').to(
            device='cuda')
        transformer_encoder_7 = nn.TransformerEncoder(encoder_layer_7, num_layers=1, norm=None).to(device='cuda')
        encoder_layer_8 = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=1, dim_feedforward=64, activation='gelu').to(
            device='cuda')
        transformer_encoder_8 = nn.TransformerEncoder(encoder_layer_8, num_layers=1, norm=None).to(device='cuda')


        #训练transformer
        x1r_output = transformer_encoder_1(x1r)
        # x1r_output = self.layernorm(x1r_output)
        x1r_output = self.denselayer(x1r_output)
        print('cls', x1r_output.shape)  # (b,x,c) SA(100, 25, 204)
        var = torch.var(x1r_output)

        x2r_output = transformer_encoder_2(x2r)
        # x2r_output = self.layernorm(x2r_output)
        x2r_output = self.denselayer(x2r_output)

        x3r_output = transformer_encoder_3(x3r)
        # x3r_output = self.layernorm(x3r_output)
        x3r_output = self.denselayer(x3r_output)

        x4r_output = transformer_encoder_4(x4r)
        # x4r_output = self.layernorm(x4r_output)
        x4r_output = self.denselayer(x4r_output)

        x5r_output = transformer_encoder_5(x5r)
        # x5r_output = self.layernorm(x5r_output)
        x5r_output = self.denselayer(x5r_output)

        x6r_output = transformer_encoder_6(x6r)
        # x6r_output = self.layernorm(x6r_output)
        x6r_output = self.denselayer(x6r_output)

        x7r_output = transformer_encoder_7(x7r)
        # x7r_output = self.layernorm(x7r_output)
        x7r_output = self.denselayer(x7r_output)

        x8r_output = transformer_encoder_8(x8r)
        # x8r_output = self.layernorm(x8r_output)
        x8r_output = self.denselayer(x8r_output)
        print('1111', x1r_output.shape) #(x,b,c) SA(25,100,self.embed_dim)

        #提取中间像素信息

        x1r_output_central = x1r_output[0, :, :] + x1r_output[13,:,:]
        x1r_output_clstoken = x1r_output[0, :, :]
        x1r_output_centraltoken = x1r_output[13, :, :]
        x1r_output_mean = reduce(x1r_output, 'x b c -> b c', reduction='mean')
        print('1111',x1r_output_central.shape) #(b,c) SA(100, 204)\

        x2r_output_central = x2r_output[0, :, :] + x2r_output[13,:,:]
        x2r_output_clstoken = x2r_output[0, :, :]
        x2r_output_centraltoken = x2r_output[13,:,:]

        x3r_output_central = x3r_output[0, :, :] + x3r_output[13,:,:]
        x3r_output_clstoken = x3r_output[0, :, :]
        x3r_output_centraltoken = x3r_output[13,:,:]

        x4r_output_central = x4r_output[0, :, :] + x4r_output[13,:,:]
        x4r_output_clstoken = x4r_output[0, :, :]
        x4r_output_centraltoken = x4r_output[13,:,:]

        x5r_output_central = x5r_output[0, :, :] + x5r_output[13,:,:]
        x5r_output_clstoken = x5r_output[0, :, :]
        x5r_output_centraltoken = x5r_output[13,:,:]

        x6r_output_central = x6r_output[0, :, :] + x6r_output[13,:,:]
        x6r_output_clstoken = x6r_output[0, :, :]
        x6r_output_centraltoken = x6r_output[13,:,:]

        x7r_output_central = x7r_output[0, :, :] + x7r_output[13,:,:]
        x7r_output_clstoken = x7r_output[0, :, :]
        x7r_output_centraltoken = x7r_output[13,:,:]

        x8r_output_central = x8r_output[0, :, :] + x8r_output[13,:,:]
        x8r_output_clstoken = x8r_output[0, :, :]
        x8r_output_centraltoken = x8r_output[13,:,:]

        #扩展维度准备合并
        x1r_output_central = rearrange(x1r_output_central, 'b c -> () b c')
        # print('1111',x1r_output_central.shape)
        x2r_output_central = rearrange(x2r_output_central, 'b c -> () b c')
        x3r_output_central = rearrange(x3r_output_central, 'b c -> () b c')
        x4r_output_central = rearrange(x4r_output_central, 'b c -> () b c')
        x5r_output_central = rearrange(x5r_output_central, 'b c -> () b c')
        x6r_output_central = rearrange(x6r_output_central, 'b c -> () b c')
        x7r_output_central = rearrange(x7r_output_central, 'b c -> () b c')
        x8r_output_central = rearrange(x8r_output_central, 'b c -> () b c')

        '只用一个扫描 or 2 or 4 '
        # preds_onedirection = torch.cat([x1r_output_centraltoken+x1r_output_clstoken,x2r_output_centraltoken+x2r_output_clstoken,x3r_output_centraltoken+x3r_output_clstoken,x4r_output_centraltoken+x4r_output_clstoken
        #                                 ,x5r_output_centraltoken+x5r_output_clstoken,x6r_output_centraltoken+x6r_output_clstoken,x7r_output_centraltoken+x7r_output_clstoken,x8r_output_centraltoken+x8r_output_clstoken], dim=1)
        # preds_onedirection = x1r_output_clstoken+x2r_output_clstoken+x3r_output_clstoken+x4r_output_clstoken
        #                                    +x5r_output_clstoken+x6r_output_clstoken+x7r_output_clstoken+x8r_output_clstoken
        x1r_output_conca = torch.cat([x1r_output_clstoken,x1r_output_centraltoken],dim=1)
        x7r_output_conca = torch.cat([x7r_output_clstoken,x7r_output_centraltoken],dim=1)
        x2r_output_conca = torch.cat([x2r_output_clstoken,x2r_output_centraltoken],dim=1)
        x8r_output_conca = torch.cat([x8r_output_clstoken,x8r_output_centraltoken],dim=1)
        x3r_output_conca = torch.cat([x3r_output_clstoken, x3r_output_centraltoken], dim=1)
        x4r_output_conca = torch.cat([x4r_output_clstoken, x4r_output_centraltoken], dim=1)
        x5r_output_conca = torch.cat([x5r_output_clstoken, x5r_output_centraltoken], dim=1)
        x6r_output_conca = torch.cat([x6r_output_clstoken, x6r_output_centraltoken], dim=1)

        x1r_output_conca = repeat(x1r_output_conca, 'b c -> b c ()')
        x7r_output_conca = repeat(x7r_output_conca, 'b c -> b c ()')
        x2r_output_conca = repeat(x2r_output_conca, 'b c -> b c ()')
        x8r_output_conca = repeat(x8r_output_conca, 'b c -> b c ()')
        x3r_output_conca = repeat(x3r_output_conca, 'b c -> b c ()')
        x4r_output_conca = repeat(x4r_output_conca, 'b c -> b c ()')
        x5r_output_conca = repeat(x5r_output_conca, 'b c -> b c ()')
        x6r_output_conca = repeat(x6r_output_conca, 'b c -> b c ()')

        preds_onedirection = torch.cat([x1r_output_conca,x2r_output_conca,x3r_output_conca,x4r_output_conca,x5r_output_conca,x6r_output_conca,x7r_output_conca,x8r_output_conca],dim=2)


        # print('```',preds_onedirection.shape)
        #
        preds_onedirection = preds_onedirection.view(preds_onedirection.size(0), -1)
        preds_onedirection = self.transformer_bn_scheme1(preds_onedirection)
        preds_onedirection = self.relu(preds_onedirection)
        preds_onedirection = self.dropout(preds_onedirection)
        preds_onedirection = self.fc_transformer_scheme1(preds_onedirection)

        "用LSTM"
        # h0_x1r = torch.zeros(1, x1r.size(1), 64).to(device='cuda')
        # c0_x1r = torch.zeros(1, x1r.size(1), 64).to(device="cuda")
        # x1r, (hn_x1r, cn_x1r) = self.lstm_2_1(x1r, (h0_x1r, c0_x1r))
        # # x1r = self.gru_2_1(x1r)[0]
        # print('x1r', x1r.shape)  #(x,b,c) (25,100,64)
        # x1r_laststep = x1r[-1]
        # x1r_laststep = self.relu(x1r_laststep)
        # x1r_laststep = torch.unsqueeze(x1r_laststep, dim=0)
        #
        # h0_x2r = torch.zeros(1, x2r.size(1), 64).to(device='cuda')
        # c0_x2r = torch.zeros(1, x2r.size(1), 64).to(device="cuda")
        # x2r, (hn_x2r, cn_x2r) = self.lstm_2_2(x2r, (h0_x2r, c0_x2r))
        # # x2r = self.gru_2_2(x2r)[0] #把x1r经过RNN的值，作为x2r的输入
        # x2r_laststep = x2r[-1]
        # x2r_laststep = self.relu(x2r_laststep)
        # x2r_laststep = torch.unsqueeze(x2r_laststep, dim=0)
        #
        # h0_x3r = torch.zeros(1, x3r.size(1), 64).to(device='cuda')
        # c0_x3r = torch.zeros(1, x3r.size(1), 64).to(device="cuda")
        # x3r, (hn_x3r, cn_x3r) = self.lstm_2_3(x3r, (h0_x3r, c0_x3r))
        # # x3r = self.gru_2_3(x3r)[0]
        # x3r_laststep = x3r[-1]
        # x3r_laststep = self.relu(x3r_laststep)
        # x3r_laststep = torch.unsqueeze(x3r_laststep, dim=0)
        #
        # h0_x4r = torch.zeros(1, x4r.size(1), 64).to(device='cuda')
        # c0_x4r = torch.zeros(1, x4r.size(1), 64).to(device="cuda")
        # x4r, (hn_x4r, cn_x4r) = self.lstm_2_4(x4r, (h0_x4r, c0_x4r))
        # # x4r = self.gru_2_4(x4r)[0]
        # x4r_laststep = x4r[-1]
        # x4r_laststep = self.relu(x4r_laststep)
        # x4r_laststep = torch.unsqueeze(x4r_laststep, dim=0)
        #
        # h0_x5r = torch.zeros(1, x5r.size(1), 64).to(device='cuda')
        # c0_x5r = torch.zeros(1, x5r.size(1), 64).to(device="cuda")
        # x5r, (hn_x5r, cn_x5r) = self.lstm_2_5(x5r, (h0_x5r, c0_x5r))
        # # x5r = self.gru_2_5(x5r)[0]
        # x5r_laststep = x5r[-1]
        # x5r_laststep = self.relu(x5r_laststep)
        # x5r_laststep = torch.unsqueeze(x5r_laststep, dim=0)
        #
        # h0_x6r = torch.zeros(1, x6r.size(1), 64).to(device='cuda')
        # c0_x6r = torch.zeros(1, x6r.size(1), 64).to(device="cuda")
        # x6r, (hn_x6r, cn_x6r) = self.lstm_2_6(x6r, (h0_x6r, c0_x6r))
        # # x6r = self.gru_2_6(x6r)[0]
        # x6r_laststep = x6r[-1]
        # x6r_laststep = self.relu(x6r_laststep)
        # x6r_laststep = torch.unsqueeze(x6r_laststep, dim=0)
        #
        # h0_x7r = torch.zeros(1, x7r.size(1), 64).to(device='cuda')
        # c0_x7r = torch.zeros(1, x7r.size(1), 64).to(device="cuda")
        # x7r, (hn_x7r, cn_x7r) = self.lstm_2_7(x7r, (h0_x7r, c0_x7r))
        # # x7r = self.gru_2_7(x7r)[0]
        # x7r_laststep = x7r[-1]
        # x7r_laststep = self.relu(x7r_laststep)
        # x7r_laststep = torch.unsqueeze(x7r_laststep, dim=0)
        #
        # h0_x8r = torch.zeros(1, x8r.size(1), 64).to(device='cuda')
        # c0_x8r = torch.zeros(1, x8r.size(1), 64).to(device="cuda")
        # x8r, (hn_x8r, cn_x8r) = self.lstm_2_8(x8r, (h0_x8r, c0_x8r))
        # # x8r = self.gru_2_8(x8r)[0]
        # x8r_laststep = x8r[-1]
        # x8r_laststep = self.relu(x8r_laststep)
        # x8r_laststep = torch.unsqueeze(x8r_laststep, dim=0)
        # print('x8r_last',x8r_laststep.shape)


        "scheme 1"
        # x_strategy_FLC = torch.cat([x1r_output_central,x2r_output_central,x3r_output_central,x4r_output_central,x5r_output_central,x6r_output_central,x7r_output_central,x8r_output_central],dim=0)
        # print('x_strategy_FLC', x_strategy_FLC.shape) # (x, b, c) (8 , batch, 64)
        # x_strategy_FLC = rearrange(x_strategy_FLC, 'x b c -> b x c')
        # x_strategy_FLC += self.pos_embedding_conca_scheme1[:, :(x_strategy_FLC.size(1))]
        # x_strategy_FLC = rearrange(x_strategy_FLC, 'b x c -> x b c')
        # # 设置transformer的参数给8个direction
        # encoder_layer_conca = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=1, dim_feedforward=32, activation='gelu').to(
        #     device='cuda')
        # transformer_encoder_conca = nn.TransformerEncoder(encoder_layer_conca, num_layers=1, norm=None).to(device='cuda')
        #
        # x_strategy_FLC_output = transformer_encoder_conca(x_strategy_FLC)
        # x_strategy_FLC_output = self.denselayer_scheme1(x_strategy_FLC_output)
        # x_strategy_FLC_output = reduce(x_strategy_FLC_output, 'x b c -> 1 b c', reduction='mean')
        # x_strategy_FLC_output = self.relu(x_strategy_FLC_output)
        # x_strategy_FLC_output = x_strategy_FLC_output.view(x_strategy_FLC_output.size(1), -1)
        # x_strategy_FLC_output = self.transformer_bn_scheme1(x_strategy_FLC_output)
        # x_strategy_FLC_output = self.dropout(x_strategy_FLC_output)
        # preds_scheme1 = self.fc_transformer_scheme1(x_strategy_FLC_output)

        "scheme 2"
        x1_7r = torch.cat([x1r_output_conca, x7r_output_conca], dim=1 )
        x2_8r = torch.cat([x2r_output_conca, x8r_output_conca], dim=1 )
        x3_5r = torch.cat([x3r_output_conca, x5r_output_conca], dim=1 )
        x4_6r = torch.cat([x4r_output_conca, x6r_output_conca], dim=1 )
        x_strategy_FLC = torch.cat([x1_7r, x2_8r, x3_5r, x4_6r], dim=2)
        x_strategy_FLC = rearrange(x_strategy_FLC, 'b c x -> b x c')
        print('...', x_strategy_FLC.shape)
        cls_tokens_FLC = repeat(self.cls_token_FLC, '() n c -> b n c', b=b)
        x_strategy_FLC = torch.cat((cls_tokens_FLC, x_strategy_FLC), dim=1)
        x_strategy_FLC += self.pos_embedding_conca_scheme2[:, :(x_strategy_FLC.size(1) + 1 )]
        x_strategy_FLC = rearrange(x_strategy_FLC, 'b x c -> x b c')
        # 设置transformer的参数给4对directions
        encoder_layer_conca = nn.TransformerEncoderLayer(d_model=self.embed_dim * 4, nhead=1, dim_feedforward=64,
                                                         activation='gelu').to(device='cuda')
        transformer_encoder_conca = nn.TransformerEncoder(encoder_layer_conca, num_layers=1, norm=None).to(device='cuda')
        x_strategy_FLC_output = transformer_encoder_conca(x_strategy_FLC)
        x_strategy_FLC_output = self.denselayer_scheme2(x_strategy_FLC_output)

        x_strategy_FLC_output_clstoken = x_strategy_FLC_output[0,:,:]
        x_strategy_FLC_output_meantoken = (x_strategy_FLC_output[1,:,:] + x_strategy_FLC_output[2,:,:] + x_strategy_FLC_output[3,:,:] + x_strategy_FLC_output[4,:,:])
        # x_strategy_FLC_output = reduce(x_strategy_FLC_output, 'x b c -> 1 b c', reduction='mean')

        x_strategy_FLC_output = self.relu(x_strategy_FLC_output_meantoken)
        x_strategy_FLC_output = x_strategy_FLC_output.view(x_strategy_FLC_output.size(0), -1)
        x_strategy_FLC_output = self.transformer_bn_scheme2(x_strategy_FLC_output)
        x_strategy_FLC_output = self.dropout(x_strategy_FLC_output)
        preds_scheme2 = self.fc_transformer_scheme2(x_strategy_FLC_output)

        return preds_scheme2

class zhouICPR2022(nn.Module):

    @staticmethod
    def weight_init(m):
        # All weight matrices in our RNN and bias vectors are initialized with a uniform distribution, and the values of these weight matrices and bias vectors are initialized in the range [−0.1,0.1]
        if isinstance(m, (nn.Linear, nn.GRU, nn.Conv3d,nn.LSTM, nn.TransformerEncoderLayer, nn.TransformerEncoder)):
            init.kaiming_uniform_(m.weight.data)
            init.kaiming_uniform_(m.bias.data)

    def __init__(self, input_channels, n_classes, patch_size=9, embed_dim = 64):
        # The proposed network model uses a single recurrent layer that adopts our modified GRUs of size 64 with sigmoid gate activation and PRetanh activation functions for hidden representations
        super(zhouICPR2022, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        'do not use self.n_classes = n_classes'
        self.pos_embedding_81_plus_1 = nn.Parameter(torch.randn(1, 81 + 1, self.input_channels ))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding_49_plus_1 = nn.Parameter(torch.randn(1, 49+1, self.input_channels ))
        self.cls_token_49_plus_1 = nn.Parameter(torch.randn(1, 1, self.input_channels))
        self.pos_embedding_25_plus_1 = nn.Parameter(torch.randn(1, 25+1, self.input_channels ))
        self.cls_token_25_plus_1 = nn.Parameter(torch.randn(1, 1, self.input_channels))
        self.pos_embedding_9_plus_1 = nn.Parameter(torch.randn(1, 9+1, self.input_channels ))
        self.cls_token_9_plus_1 = nn.Parameter(torch.randn(1, 1, self.input_channels))
        self.cls_token_FLC = nn.Parameter(torch.randn(1, 1, embed_dim * 2))

        self.alpha = nn.Parameter(torch.randn(1,1))
        self.beta = nn.Parameter(torch.randn(1, 1))
        self.gamma = nn.Parameter(torch.randn(1, 1))

        self.conv_11to9 = nn.Conv2d(in_channels=input_channels,out_channels=input_channels ,kernel_size=3)
        self.encoder_layer_81_plus_1 = nn.TransformerEncoderLayer(d_model=self.input_channels , nhead=1, dim_feedforward=64, activation='gelu')
        self.transformer_encoder_81_plus_1 = nn.TransformerEncoder(self.encoder_layer_81_plus_1, num_layers=1,norm=None)


        self.conv_9to7 = nn.Conv2d(in_channels=self.input_channels , out_channels=self.input_channels , kernel_size=3)
        self.encoder_layer_49_plus_1 = nn.TransformerEncoderLayer(d_model=self.input_channels , nhead=1, dim_feedforward=64,activation='gelu')
        self.transformer_encoder_49_plus_1 = nn.TransformerEncoder(self.encoder_layer_49_plus_1, num_layers=1, norm=None)
        self.deconv_1to7 = nn.ConvTranspose2d(in_channels=input_channels,out_channels=input_channels,kernel_size=6,stride=1)

        self.conv_7to5 = nn.Conv2d(in_channels=self.input_channels ,out_channels=self.input_channels ,kernel_size=3)
        self.encoder_layer_25_plus_1 = nn.TransformerEncoderLayer(d_model=self.input_channels , nhead=1, dim_feedforward=64,activation='gelu')
        self.transformer_encoder_25_plus_1 = nn.TransformerEncoder(self.encoder_layer_25_plus_1, num_layers=1, norm=None)
        self.deconv_1to5 = nn.ConvTranspose2d(in_channels=input_channels,out_channels=input_channels,kernel_size=4,stride=1)

        self.conv_5to3 = nn.Conv2d(in_channels=self.input_channels , out_channels=self.input_channels , kernel_size=3)
        self.encoder_layer_9_plus_1 = nn.TransformerEncoderLayer(d_model=self.input_channels , nhead=1, dim_feedforward=64,activation='gelu')
        self.transformer_encoder_9_plus_1 = nn.TransformerEncoder(self.encoder_layer_9_plus_1, num_layers=1, norm=None)
        self.deconv_1to3 = nn.ConvTranspose2d(in_channels=input_channels,out_channels=input_channels, kernel_size=2,stride=1)

        self.conv_3to1 = nn.Conv2d(in_channels=self.input_channels , out_channels=self.input_channels , kernel_size=3)

        self.adapooling = nn.AdaptiveAvgPool1d(1)
        self.adapooling2d = nn.AdaptiveAvgPool2d((1,1))
        self.gru_bn = nn.BatchNorm1d(patch_size ** 2 * input_channels)
        self.lstm_bn_1 = nn.BatchNorm1d((64)*1)
        self.lstm_bn_1_2 = nn.BatchNorm1d((64) * (patch_size)**2)
        self.lstm_bn_2 = nn.BatchNorm1d((64)*8)
        self.lstm_bn_2_2 = nn.BatchNorm1d((64) * 8 * patch_size**2)
        self.gru_bn_2 = nn.BatchNorm1d((patch_size ** 2) * (patch_size**2))
        self.gru_bn_3 = nn.BatchNorm1d((patch_size ** 2) * (patch_size ** 2) * 2)
        self.gru_bn_4 = nn.BatchNorm1d(8 * 64)
        self.gru_bn_laststep = nn.BatchNorm1d(64)
        self.transformer_bn_scheme1 = nn.BatchNorm1d(embed_dim)
        self.transformer_bn_scheme2 = nn.BatchNorm1d(embed_dim * 2)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(input_channels *3 , n_classes)
        self.bn = nn.BatchNorm1d(input_channels *3  )
        self.lstm_fc_1 = nn.Linear(64 * 1, n_classes)
        self.lstm_fc_1_2 = nn.Linear(64 * (patch_size**2), n_classes)
        self.lstm_fc_2 = nn.Linear(64*8,n_classes)
        self.lstm_fc_2_2 = nn.Linear(64 * 8 * patch_size**2, n_classes)
        self.fc_2 = nn.Linear((patch_size ** 2) * (patch_size**2), n_classes)
        self.fc_3 = nn.Linear((patch_size ** 2) * (patch_size ** 2) * 2, n_classes)
        self.fc_4 = nn.Linear(8 * 64, n_classes)
        self.layernorm = nn.LayerNorm(embed_dim)
        self.denselayer = nn.Linear(input_channels, input_channels)
        self.denselayer1 = nn.Linear(input_channels * 2, input_channels * 2)
        self.denselayer2 = nn.Linear(input_channels * 3, input_channels * 3)
        self.denselayer3 = nn.Linear(input_channels * 4, input_channels * 4)
        self.denselayer4 = nn.Linear(input_channels * 5, input_channels * 5)
        self.denselayer_scheme1 = nn.Linear(embed_dim,embed_dim)
        self.denselayer_scheme2 = nn.Linear(embed_dim * 2, embed_dim * 2)
        self.fc_laststep = nn.Linear(64, n_classes)
        self.fc_transformer_scheme1 = nn.Linear(embed_dim,n_classes)
        self.fc_transformer_scheme2 = nn.Linear(embed_dim * 2 , n_classes)
        self.linearembedding = nn.Linear(in_features=input_channels,out_features=embed_dim)
        self.softmax = nn.Softmax()
        self.norm = nn.LayerNorm(input_channels)
        self.cos = nn.CosineSimilarity(dim=1)
        self.distance = nn.PairwiseDistance(p=2)


    def forward(self, x): #初始是第1方向
        print('x.shape1',x.shape)

        # x_11by11 = x.squeeze(1)
        # '11*11 --conv--> 9*9'
        # x_9by9 = self.conv_11to9(x_11by11)
        # x_9by9_flat = rearrange(x_9by9, 'b c h w -> b (h w) c')
        # b, n, c = x_9by9_flat.shape
        # x_9by9_flat += self.pos_embedding_81_plus_1[:, :(x_9by9_flat.size(1))]
        # print('x_9by9_flat_plus_cls', x_9by9_flat.shape)
        # x_9by9_flat_plus_cls = rearrange(x_9by9_flat, 'b n c -> n b c')
        # x_9by9_flat_plus_cls_output = self.transformer_encoder_81_plus_1(x_9by9_flat_plus_cls)
        # x_9by9_flat_plus_cls_output = self.denselayer(x_9by9_flat_plus_cls_output)
        # print('x_9by9_flat_plus_cls_output', x_9by9_flat_plus_cls_output.shape)
        # central_token = x_9by9_flat_plus_cls_output[40, :, :]
        # central_token = repeat(central_token, 'b c -> b () c')
        # central_token = repeat(central_token, 'b () c -> b c 9 9')
        # x_9by9 = central_token

        x_9by9 = x.squeeze(1)
        '9*9 --conv--> 7*7'
        x_7by7 = self.conv_9to7(x_9by9)
        x_7by7_base = x_7by7
        x_7by7_flat = rearrange(x_7by7, 'b c h w -> b (h w) c')
        #cls token + positional embeding
        b,n,c = x_7by7_flat.shape
        cls_tokens_49_plus_1 = repeat(self.cls_token_49_plus_1, '() n c -> b n c', b=b)
        x_7by7_flat_plus_cls = torch.cat((cls_tokens_49_plus_1, x_7by7_flat), dim=1)
        # print('x_7by7_flat_plus_cls',x_7by7_flat_plus_cls.shape)

        x_7by7_flat_plus_cls += self.pos_embedding_49_plus_1[:, :(x_7by7_flat_plus_cls.size(1) + 1)]
        print('x_7by7_flat_plus_cls',x_7by7_flat_plus_cls.shape)
        # x_7by7_flat_plus_cls = self.dropout(x_7by7_flat_plus_cls)
        #trans setting

        # train trans
        x_7by7_flat_plus_cls = rearrange(x_7by7_flat, 'b n c -> n b c')
        x_7by7_flat_plus_cls_output = self.transformer_encoder_49_plus_1(x_7by7_flat_plus_cls)
        # plt.imshow(x_7by7_flat_plus_cls_output[:, 0, :].cpu().detach().numpy())
        x_7by7_flat_plus_cls_output = self.denselayer(x_7by7_flat_plus_cls_output)
        cls_token_7by7 = x_7by7_flat_plus_cls_output[0, :, :]
        print('x_7by7_flat_plus_cls_output',x_7by7_flat_plus_cls_output.shape) #(49+1, b, channel)
        x_7by7_flat_plus_cls_output_pooling = rearrange((x_7by7_flat_plus_cls_output - x_7by7_flat_plus_cls_output[0,:,:]), 'l n c -> n c l ')
        # plt.show()
        # plt.plot(cls_token_7by7[0, :].cpu().detach().numpy(), color='orange')
        central_token_7by7 = x_7by7_flat_plus_cls_output[25,:,:]
        # plt.plot(central_token_7by7[0, :].cpu().detach().numpy(), color='red')
        pooling_token_7by7 = self.adapooling(x_7by7_flat_plus_cls_output_pooling)
        pooling_token_7by7 = reduce(pooling_token_7by7, 'b c 1 -> b c', reduction='mean')
        # plt.plot(pooling_token_7by7[0, :].cpu().detach().numpy(), color='blue')
        # plt.show()
        print('pooling',pooling_token_7by7.shape)
        print('cls:',cls_token_7by7.shape)
        print('central:', central_token_7by7.shape)
        print("cls_pooling 1:", self.cos(cls_token_7by7[:,:],pooling_token_7by7[:,:]), "cls_central:",self.cos(cls_token_7by7[:,:],central_token_7by7[:,:]), "pooling_central", self.cos(pooling_token_7by7[:,:],central_token_7by7[:,:]))

        #adaptive pooling
        # cls_token_7by7 = repeat(cls_token_7by7, 'b c -> b () c')
        # central_token_7by7 = repeat(central_token_7by7, 'b c -> b () c')
        # print('cls:', cls_token_49_plus_1_output.shape)
        # print('central:', central_token.shape)
        # cls_token_49_plus_1_output = self.adapooling(cls_token_49_plus_1_output)
        # central_token = self.adapooling(central_token)
        # print('cls:', cls_token_49_plus_1_output.shape)
        print('central:', central_token_7by7.shape)
        # cls_token_49_plus_1_output = repeat(cls_token_49_plus_1_output, 'b () c -> b c 7 7') #先试试直接变成7by7
        # central_token = repeat(central_token, 'b () c -> b c 7 7')
        # central_token = self.deconv_1to7(central_token)
        # print('cls:', cls_token_49_plus_1_output.shape)
        # print('deconv_central:', central_token.shape)
        #重建成7by7的patch
        x_7by7 = x_7by7_base
        # x_7by7 = torch.cat((cls_token_49_plus_1_output,central_token), dim=1)
        print('x_7by7',x_7by7.shape)

        '7*7 --conv--> 5*5'
        x_5by5 = self.conv_7to5(x_7by7)
        x_5by5_base = x_5by5
        x_5by5_flat = rearrange(x_5by5, 'b c h w -> b (h w) c')
        # cls token + positional embeding
        b, n, c = x_5by5_flat.shape
        cls_tokens_25_plus_1 = repeat(self.cls_token_25_plus_1, '() n c -> b n c', b=b)
        x_5by5_flat_plus_cls = torch.cat((cls_tokens_25_plus_1, x_5by5_flat), dim=1)
        # print('x_5by5_flat_plus_cls:', x_5by5_flat_plus_cls.shape)

        x_5by5_flat_plus_cls += self.pos_embedding_25_plus_1[:, :(x_5by5_flat_plus_cls.size(1) + 1)]
        # print('x_5by5_flat_plus_cls:', x_5by5_flat_plus_cls.shape)
        # x_5by5_flat_plus_cls = self.dropout(x_5by5_flat_plus_cls)
        # trans setting
        # train trans
        x_5by5_flat_plus_cls = rearrange(x_5by5_flat_plus_cls, 'b n c -> n b c')
        x_5by5_flat_plus_cls_output = self.transformer_encoder_25_plus_1(x_5by5_flat_plus_cls)
        x_5by5_flat_plus_cls_output = self.denselayer(x_5by5_flat_plus_cls_output)
        print('x_5by5_flat_plus_cls_output', x_5by5_flat_plus_cls_output.shape)  # (25+1, b, channel)
        # x_5by5_flat_plus_cls_output_pooling = rearrange(
        #     (x_5by5_flat_plus_cls_output - x_5by5_flat_plus_cls_output[0, :, :]), 'l n c -> n c l ')
        pooling_token_5by5 = reduce(
            x_5by5_flat_plus_cls_output, 'l n c -> n c ',reduction='mean')
        # plt.imshow(x_7by7_flat_plus_cls_output.cpu()[:,0,:])
        # plt.show()
        cls_token_5by5 = x_5by5_flat_plus_cls_output[0, :, :] + self.softmax(cls_token_7by7)
        # plt.plot(cls_token_5by5[0, :].cpu().detach().numpy(), color='orange')
        central_token_5by5 = x_5by5_flat_plus_cls_output[12, :, :] + self.softmax(central_token_7by7)
        # plt.plot(central_token_5by5[0, :].cpu().detach().numpy(), color='red')
        # pooling_token_5by5 = self.adapooling(x_5by5_flat_plus_cls_output_pooling)
        # pooling_token_5by5 = reduce(pooling_token_5by5, 'b c 1 -> b c', reduction='mean')
        pooling_token_5by5 = pooling_token_5by5 + self.softmax(pooling_token_7by7)
        # plt.plot(pooling_token_5by5[0, :].cpu().detach().numpy(), color='blue')
        # plt.show()
        print('cls:', cls_token_5by5.shape)
        print('central:', central_token_5by5.shape)
        print("cls_pooling 2:", self.cos(cls_token_5by5[:,:],pooling_token_5by5[:,:]), "cls_central:",self.cos(cls_token_5by5[:,:],central_token_5by5[:,:]), "pooling_central", self.cos(pooling_token_5by5[:,:],central_token_5by5[:,:]))

        # adaptive pooling
        # cls_token_25_plus_1_output = repeat(cls_token_25_plus_1_output, 'b c -> b () c')
        # central_token = repeat(central_token, 'b c -> b () c')
        # print('cls:', cls_token_25_plus_1_output.shape)
        # print('central:', central_token.shape)
        # adpooling = nn.AdaptiveAvgPool1d(output_size=102).to(
        #     device='cuda')  # output_size根据不同的数据集,需要改动 一半的self.input_channels
        # cls_token_25_plus_1_output = self.adapooling(cls_token_25_plus_1_output)
        # central_token = self.adapooling(central_token)
        # print('cls:', cls_token_25_plus_1_output.shape)
        # print('central:', central_token.shape)
        # cls_token_25_plus_1_output = repeat(cls_token_25_plus_1_output, 'b () c -> b c 5 5')  # 先试试直接变成5by5
        # central_token = repeat(central_token, 'b () c -> b c 5 5')
        # central_token = self.deconv_1to5(central_token)
        # print('cls:', cls_token_25_plus_1_output.shape)
        # print('central:', central_token.shape)
        # 重建成5by5的patch
        x_5by5 = x_5by5_base
        # x_5by5 = torch.cat((cls_token_25_plus_1_output, central_token), dim=1)
        print('x_5by5', x_5by5.shape)
        #
        '5*5 --conv--> 3*3'

        x_3by3 = self.conv_5to3(x_5by5)
        x_3by3_base = x_3by3
        x_3by3_flat = rearrange(x_3by3, 'b c h w -> b (h w) c')
        # cls token + positional embeding

        b, n, c = x_3by3_flat.shape
        cls_tokens_9_plus_1 = repeat(self.cls_token_9_plus_1, '() n c -> b n c', b=b)
        x_3by3_flat_plus_cls = torch.cat((cls_tokens_9_plus_1, x_3by3_flat), dim=1)
        # print('x_3by3_flat_plus_cls:', x_3by3_flat_plus_cls.shape)

        x_3by3_flat_plus_cls += self.pos_embedding_9_plus_1[:, :(x_3by3_flat_plus_cls.size(1) + 1)]
        # print('x_3by3_flat_plus_cls:', x_3by3_flat_plus_cls.shape)
        # x_3by3_flat_plus_cls = self.dropout(x_3by3_flat)
        # trans setting
        # train trans
        x_3by3_flat_plus_cls = rearrange(x_3by3_flat_plus_cls, 'b n c -> n b c')
        x_3by3_flat_plus_cls_output = self.transformer_encoder_9_plus_1(x_3by3_flat_plus_cls)
        x_3by3_flat_plus_cls_output = self.denselayer(x_3by3_flat_plus_cls_output)
        print('x_3by3_flat_plus_cls_output', x_3by3_flat_plus_cls_output.shape)  # (9+1, b, channel)
        # x_3by3_flat_plus_cls_output_pooling = reduce((x_3by3_flat_plus_cls_output, 'n b c -> b c'), reduction='mean')
        pooling_token_3by3 = reduce(x_3by3_flat_plus_cls_output, 'n b c -> b c', reduction='mean')
        # plt.imshow(x_7by7_flat_plus_cls_output.cpu()[:,0,:])
        # plt.show()
        # plt.plot(x_3by3_flat_plus_cls_output.permute(2,1,0)[:,0,:].cpu().detach().numpy())
        cls_token_3by3 = x_3by3_flat_plus_cls_output[0, :, :] + self.softmax(cls_token_5by5) + self.softmax(cls_token_7by7)
        # plt.plot(cls_token_3by3[0, :].cpu().detach().numpy(), color='black',linewidth=3)
        central_token_3by3 = x_3by3_flat_plus_cls_output[4, :, :] + self.softmax(central_token_5by5) + self.softmax(central_token_7by7)
        # plt.plot(central_token_3by3[0, :].cpu().detach().numpy(), color='red', linewidth=3)
        # pooling_token_3by3 = self.adapooling(x_3by3_flat_plus_cls_output_pooling)
        # pooling_token_3by3 = reduce(pooling_token_3by3, 'b c 1 -> b c', reduction='mean')
        pooling_token_3by3 = pooling_token_3by3 + self.softmax(pooling_token_7by7) + self.softmax(pooling_token_5by5)
        # plt.plot(pooling_token_3by3[0, :].cpu().detach().numpy(), color='blue',linewidth=3)

        # plt.grid(linewidth=0.5, color='black')
        # plt.title('All tokens', fontdict={'size': 40})
        # plt.xlabel('Spectral size', fontdict={'size': 40}, fontweight='bold')
        # plt.ylabel('Values', fontdict={'size': 40})
        # plt.xticks(fontsize=35)
        # plt.yticks(fontsize=35)
        # bwith = 2  # 边框宽度设置为2
        # TK = plt.gca()  # 获取边框
        # TK.spines['bottom'].set_linewidth(bwith)  # 图框下边
        # TK.spines['left'].set_linewidth(bwith)  # 图框左边
        # TK.spines['top'].set_linewidth(bwith)  # 图框上边
        # TK.spines['right'].set_linewidth(bwith)  # 图框右边

        # plt.show()
        print("cls_pooling 3:", self.cos(cls_token_3by3[:,:],pooling_token_3by3[:,:]), "cls_central:",self.cos(cls_token_3by3[:,:],central_token_3by3[:,:]), "pooling_central", self.cos(pooling_token_3by3[:,:],central_token_3by3[:,:]))

        # print('cls:', cls_token_9_plus_1_output.shape)
        # print('central:', central_token.shape)
        # adaptive pooling
        # cls_token_9_plus_1_output = repeat(cls_token_9_plus_1_output, 'b c -> b () c')
        # central_token = repeat(central_token, 'b c -> b () c')
        # print('cls:', cls_token_9_plus_1_output.shape)
        # print('central:', central_token.shape)
        # adpooling = nn.AdaptiveAvgPool1d(output_size=102).to(
        #     device='cuda')  # output_size根据不同的数据集,需要改动 一半的self.input_channels
        # cls_token_9_plus_1_output = self.adapooling(cls_token_9_plus_1_output)
        # central_token = self.adapooling(central_token)
        # print('cls:', cls_token_9_plus_1_output.shape)
        # print('central:', central_token.shape)
        # cls_token_9_plus_1_output = repeat(cls_token_9_plus_1_output, 'b () c -> b c 3 3')  # 先试试直接变成5by5
        # central_token = repeat(central_token, 'b () c -> b c 3 3')
        # central_token = self.deconv_1to3(central_token)
        # print('cls:', cls_token_9_plus_1_output.shape)
        print('central:', central_token_3by3.shape)
        # 重建成3by3的patch
        x_3by3 = x_3by3_base
        # x_3by3 = torch.cat((cls_token_9_plus_1_output, central_token), dim=1)
        print('x_3by3', x_3by3.shape)

        #
        "3*3 --conv--> 1*1"
        x_1by1 = self.conv_3to1(x_3by3)
        x_1by1_base = self.conv_3to1(x_3by3)
        print('x_1by1:',x_1by1.shape)
        x_1by1 = rearrange(x_1by1, 'b c () () -> b c')
        print('x_1by1:', x_1by1.shape)
        central_token_1by1 = x_1by1
        cls_token_1by1 = x_1by1
        pooling_token_1by1 = self.adapooling2d(x_1by1_base)
        pooling_token_1by1 = reduce(pooling_token_1by1, 'b c 1 1 -> b c', reduction='mean')
        #之前是直接加起来，
        central_token_all = central_token_1by1 + self.softmax(central_token_3by3) + self.softmax(central_token_5by5) + self.softmax(central_token_7by7)
        cls_token_all = cls_token_1by1 + self.softmax(cls_token_3by3) + self.softmax(cls_token_5by5) + self.softmax(cls_token_7by7)
        pooling_token_all = pooling_token_1by1 + self.softmax(pooling_token_3by3) + self.softmax(pooling_token_5by5) + self.softmax(pooling_token_7by7)
        print('central_token_all', central_token_all.shape)
        # plt.plot(cls_token_all[0,:].cpu().detach().numpy(),color = 'orange')
        # plt.plot(central_token_all[0, :].cpu().detach().numpy(), color='red')
        # plt.plot(pooling_token_all[0,:].cpu().detach().numpy(), color='blue')
        # plt.show()
        alpha = self.sigmoid(repeat(self.alpha, '1 1 -> b 1',b=b))
        beta = self.sigmoid(repeat(self.beta, '1 1 -> b 1',b=b))
        gamma = 1 - beta

        print("cls_pooling 4:", self.cos(cls_token_all[:,:],pooling_token_all[:,:]), "cls_central:",self.cos(cls_token_all[:,:],central_token_all[:,:]), "pooling_central", self.cos(pooling_token_all[:,:],central_token_all[:,:]))
        token_conca_all = torch.cat([alpha * cls_token_all, beta * central_token_all, gamma * pooling_token_all],dim=1)
        token_conca_all = self.bn(token_conca_all)
        token_conca_all = self.dropout(token_conca_all)
        preds = self.fc(token_conca_all)
        print('alpha:', alpha, 'beta:', beta, 'gamma:', gamma )
        # x_1by1 = self.bn(cls_token_all +central_token_all+pooling_token_all)
        # x_1by1 = self.dropout(x_1by1)
        # preds2 = self.fc(x_1by1)


        "scheme 1"
        # x_strategy_FLC = torch.cat([x1r_output_central,x2r_output_central,x3r_output_central,x4r_output_central,x5r_output_central,x6r_output_central,x7r_output_central,x8r_output_central],dim=0)
        # print('x_strategy_FLC', x_strategy_FLC.shape) # (x, b, c) (8 , batch, 64)
        # x_strategy_FLC = rearrange(x_strategy_FLC, 'x b c -> b x c')
        # x_strategy_FLC += self.pos_embedding_conca_scheme1[:, :(x_strategy_FLC.size(1))]
        # x_strategy_FLC = rearrange(x_strategy_FLC, 'b x c -> x b c')
        # # 设置transformer的参数给8个direction
        # encoder_layer_conca = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=1, dim_feedforward=32, activation='gelu').to(
        #     device='cuda')
        # transformer_encoder_conca = nn.TransformerEncoder(encoder_layer_conca, num_layers=1, norm=None).to(device='cuda')
        #
        # x_strategy_FLC_output = transformer_encoder_conca(x_strategy_FLC)
        # x_strategy_FLC_output = self.denselayer_scheme1(x_strategy_FLC_output)
        # x_strategy_FLC_output = reduce(x_strategy_FLC_output, 'x b c -> 1 b c', reduction='mean')
        # x_strategy_FLC_output = self.relu(x_strategy_FLC_output)
        # x_strategy_FLC_output = x_strategy_FLC_output.view(x_strategy_FLC_output.size(1), -1)
        # x_strategy_FLC_output = self.transformer_bn_scheme1(x_strategy_FLC_output)
        # x_strategy_FLC_output = self.dropout(x_strategy_FLC_output)
        # preds = self.fc_transformer_scheme1(x_strategy_FLC_output)

        "scheme 2"
        # x1_7r = torch.cat([x1r_output_central, x7r_output_central], dim=2 )
        # x2_8r = torch.cat([x2r_output_central, x8r_output_central], dim=2 )
        # x3_5r = torch.cat([x3r_output_central, x5r_output_central], dim=2 )
        # x4_6r = torch.cat([x4r_output_central, x6r_output_central], dim=2 )
        # x_strategy_FLC = torch.cat([x1_7r, x2_8r, x3_5r, x4_6r], dim=0)
        # x_strategy_FLC = rearrange(x_strategy_FLC, 'x b c -> b x c')
        # cls_tokens_FLC = repeat(self.cls_token_FLC, '() n c -> b n c', b=b)
        # x_strategy_FLC = torch.cat((cls_tokens_FLC, x_strategy_FLC), dim=1)
        # x_strategy_FLC += self.pos_embedding_conca_scheme2[:, :(x_strategy_FLC.size(1) + 1 )]
        # x_strategy_FLC = rearrange(x_strategy_FLC, 'b x c -> x b c')
        # # 设置transformer的参数给4对directions
        # encoder_layer_conca = nn.TransformerEncoderLayer(d_model=self.embed_dim * 2, nhead=1, dim_feedforward=32,
        #                                                  activation='gelu').to(device='cuda')
        # transformer_encoder_conca = nn.TransformerEncoder(encoder_layer_conca, num_layers=1, norm=None).to(device='cuda')
        # x_strategy_FLC_output = transformer_encoder_conca(x_strategy_FLC)
        # x_strategy_FLC_output = self.denselayer_scheme2(x_strategy_FLC_output)
        # x_strategy_FLC_output = reduce(x_strategy_FLC_output, 'x b c -> 1 b c', reduction='mean')
        # x_strategy_FLC_output = self.relu(x_strategy_FLC_output)
        # x_strategy_FLC_output = x_strategy_FLC_output.view(x_strategy_FLC_output.size(1), -1)
        # x_strategy_FLC_output = self.transformer_bn_scheme2(x_strategy_FLC_output)
        # x_strategy_FLC_output = self.dropout(x_strategy_FLC_output)
        # preds = self.fc_transformer_scheme2(x_strategy_FLC_output)

        return preds

class zhouEightDRNN_kamata_singleD(nn.Module):

    @staticmethod
    def weight_init(m):
        # All weight matrices in our RNN and bias vectors are initialized with a uniform distribution, and the values of these weight matrices and bias vectors are initialized in the range [−0.1,0.1]
        if isinstance(m, (nn.Linear, nn.GRU)):
            init.kaiming_uniform_(m.weight.data)
            init.kaiming_uniform_(m.bias.data)

    def __init__(self, input_channels, n_classes, patch_size=5):
        # The proposed network model uses a single recurrent layer that adopts our modified GRUs of size 64 with sigmoid gate activation and PRetanh activation functions for hidden representations
        super(zhouEightDRNN_kamata_singleD, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.gru = nn.GRU(patch_size ** 2, patch_size ** 2, 1, bidirectional=False,
                          batch_first=False)  # TODO: try to change this ?
        self.gru_2 = nn.GRU(input_channels, input_channels, 1, bidirectional=False)
        self.gru_2_1 = nn.GRU(input_channels, 64, 2, bidirectional=False)
        self.gru_2_2 = nn.GRU(input_channels, 64, 2, bidirectional=False)
        self.gru_2_3 = nn.GRU(input_channels, 64, 2, bidirectional=False)
        self.gru_2_4 = nn.GRU(input_channels, 64, 2, bidirectional=False)
        self.gru_2_5 = nn.GRU(input_channels, 64, 2, bidirectional=False)
        self.gru_2_6 = nn.GRU(input_channels, 64, 2, bidirectional=False)
        self.gru_2_7 = nn.GRU(input_channels, 64, 2, bidirectional=False)
        self.gru_2_8 = nn.GRU(input_channels, 64, 2, bidirectional=False)
        self.gru_3 = nn.GRU(input_channels, 64, 1, bidirectional=True)
        self.gru_3_1 = nn.GRU(input_channels,64, 1, bidirectional=True)
        self.gru_3_2 = nn.GRU(input_channels, 64, 1, bidirectional=True)
        self.gru_3_3 = nn.GRU(input_channels, 64, 1, bidirectional=True)
        self.gru_3_4 = nn.GRU(input_channels, 64, 1, bidirectional=True)
        self.gru_3_5 = nn.GRU(input_channels, 64, 1, bidirectional=True)
        self.gru_3_6 = nn.GRU(input_channels, 64, 1, bidirectional=True)
        self.gru_3_7 = nn.GRU(input_channels, 64, 1, bidirectional=True)
        self.gru_3_8 = nn.GRU(input_channels, 64, 1, bidirectional=True)
        self.gru_bn = nn.BatchNorm1d(patch_size ** 2 * input_channels)
        self.gru_bn_2 = nn.BatchNorm1d((patch_size ** 2) * 64)
        self.gru_bn_laststep = nn.BatchNorm1d(64)
        self.gru_bn_3 = nn.BatchNorm1d((patch_size ** 2) * (64) * 2)
        self.tanh = nn.Tanh()
        self.relu = nn.PReLU()
        self.elu = nn.ELU()
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(patch_size ** 2 * input_channels, n_classes)
        self.fc_2 = nn.Linear((patch_size ** 2) * 64, n_classes)
        self.fc_3 = nn.Linear((patch_size ** 2) * (64) * 2, n_classes)
        self.fc_laststep = nn.Linear(64, n_classes)
        self.softmax = nn.Softmax()

    def forward(self, x):  # 初始是第1方向
        x = x.squeeze(1)
        # x_matrix = x[0, :, :, :]
        # x_matrix = x_matrix.cpu()
        # plt.subplot(121)
        # plt.imshow(x_matrix[-1, :, :], interpolation='nearest', origin='upper')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # plt.xticks(fontsize=25)
        # plt.yticks(fontsize=25)
        # plt.xlabel('X-Position', fontdict={'size': 25}, fontweight='bold')
        # plt.ylabel('Y-Position', fontdict={'size': 25}, fontweight='bold')
        # plt.title('Values of last dimension in the patch', fontdict={'size': 20}, fontweight='bold')

        # 生成第1和7
        x1_0 = x[:, :, 0, :]
        x1_1 = x[:, :, 1, :]
        x1_2 = x[:, :, 2, :]
        x1_3 = x[:, :, 3, :]
        x1_4 = x[:, :, 4, :]
        x1_1f = torch.flip(x1_1, [-1, 0])
        x1_3f = torch.flip(x1_3, [-1,0])
        # plt.subplot(3, 4, 9).set_title('Spectral signatures in a patch')
        # direction_1_showpicture = torch.cat([x1_0, x1_1f, x1_2, x1_3f, x1_4], dim=2)
        # plt.xlabel('Band Numbers', fontdict={'size': 15}, fontweight='bold')
        # plt.ylabel('Spectral Values Values', fontdict={'size': 15},fontweight='bold')
        # plt.plot(direction_1_showpicture[0, :, :].cpu().detach().numpy())
        # direction_1 = torch.cat([x1_0, x1_1f, x1_2], dim=2)
        # plt.subplot(122)
        # plt.imshow(direction_1[0, :, :].cpu())
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.xlabel('Number of pixels', fontdict={'size': 20}, fontweight='bold')
        # plt.ylabel('Spectral Values', fontdict={'size': 20}, fontweight='bold')
        # plt.colorbar().ax.tick_params(labelsize=25)
        # plt.show()
        direction_1 = torch.cat([x1_0, x1_1f, x1_2,x1_3f,x1_4], dim=2)

        # plt.xlabel('Band Numbers', fontdict={'size': 35}, fontweight='bold')
        # plt.ylabel('Spectral Values Values', fontdict={'size': 35}, fontweight='bold')
        # plt.plot(direction_1[0, :, 0].cpu().detach().numpy())
        # plt.plot(direction_1[0, :, 1].cpu().detach().numpy())
        # plt.plot(direction_1[0, :, 2].cpu().detach().numpy())
        # plt.plot(direction_1[0, :, 3].cpu().detach().numpy())
        # plt.plot(direction_1[0, :, 4].cpu().detach().numpy(), linewidth=5)
        # plt.plot(direction_1[0, :, 5].cpu().detach().numpy())
        # plt.plot(direction_1[0, :, 6].cpu().detach().numpy())
        # plt.plot(direction_1[0, :, 7].cpu().detach().numpy())
        # plt.plot(direction_1[0, :, 8].cpu().detach().numpy())
        # plt.legend(['(0,0)', '(0,1)', '(0,2)', '(1,2)', '(1,1)', '(1,0)', '(2,0)', '(2,1)', '(2,2)'], prop={'size': 20},
        #            fontsize='large')
        # plt.title('Spectral signatures in a patch', fontdict={'size': 35}, fontweight='bold')
        # plt.xticks(fontsize=25)
        # plt.yticks(fontsize=25)
        # plt.show()
        #
        # print(direction_1.shape)
        # print('d1',direction_1.shape)

        direction_7 = torch.flip(direction_1, [2])

        # 生成第2和8
        x2_0 = x[:, :, :, 0]
        x2_1 = x[:, :, :, 1]
        x2_2 = x[:, :, :, 2]
        x2_3 = x[:, :, :, 3]
        x2_4 = x[:, :, :, 4]
        x2_1f = torch.flip(x2_1, [-1, 0])
        x2_3f = torch.flip(x2_3, [-1, 0])
        # direction_2 = torch.cat([x2_0, x2_1f, x2_2], dim=2)
        direction_2 = torch.cat([x2_0, x2_1f,x2_2,x2_3f,x2_4], dim=2)
        direction_8 = torch.flip(direction_2, [2])

        # 生成3和5
        x3_0 = x[:, :, 0, :]
        x3_1 = x[:, :, 1, :]
        x3_2 = x[:, :, 2, :]
        x3_3 = x[:, :, 3, :]
        x3_4 = x[:, :, 4, :]
        x3_0f = torch.flip(x3_0, [-1, 0])
        x3_2f = torch.flip(x3_2, [-1, 0])
        x3_4f = torch.flip(x3_4, [-1, 0])
        # direction_3 = torch.cat([x3_0f, x3_1, x3_2f], dim=2)
        direction_3 = torch.cat([x3_0f, x3_1, x3_2f,x3_3,x3_4f], dim=2)
        direction_5 = torch.flip(direction_3, [2])

        # 生成4和6
        x4_0 = x[:, :, :, 0]
        x4_1 = x[:, :, :, 1]
        x4_2 = x[:, :, :, 2]
        x4_3 = x[:, :, :, 3]
        x4_4 = x[:, :, :, 4]
        x4_1f = torch.flip(x4_1, [-1, 0])
        x4_3f = torch.flip(x4_3, [-1, 0])
        # direction_4 = torch.cat([x4_2, x4_1f, x4_0], dim=2)
        direction_4 = torch.cat([x4_4, x4_3f, x4_2, x4_1f, x4_0], dim=2)
        # print('d4', direction_4.shape)
        direction_6 = torch.flip(direction_4, [2])

        # 记得换成输入顺序
        # x8r = direction_8.permute(2, 0, 1)
        # x7r = direction_7.permute(2, 0, 1)
        # x6r = direction_6.permute(2, 0, 1)
        # x5r = direction_5.permute(2, 0, 1)
        # x4r = direction_4.permute(2, 0, 1)
        # x3r = direction_3.permute(2, 0, 1)
        # x2r = direction_2.permute(2, 0, 1)

        x1 = direction_1.permute(2, 0, 1) #换这里
        # x = x1
        x2 = direction_2.permute(2, 0, 1)
        x3 = direction_3.permute(2, 0, 1)
        x4 = direction_4.permute(2, 0, 1)
        x5 = direction_5.permute(2, 0, 1)
        x6 = direction_6.permute(2, 0, 1)
        x7 = direction_7.permute(2, 0, 1)
        x8 = direction_8.permute(2, 0, 1)
        # x7 = direction_7.permute(2, 0, 1)
        # print('d5.shape', x5r.shape) #25,16,200
        # plt.subplot(3, 4, 9)
        # plt.plot(direction_1[0, :, :].cpu().detach().numpy())

        # print('x1r',x1r.shape)
        x1 = self.gru_2_1(x1)[0]
        # print(x1.shape) #25,100,64
        x1r_laststep = x1[-1]
        np.save('x1_save', x1r_laststep.cpu().detach().numpy(), allow_pickle=True)
        # print(x1r_laststep.shape) #100,64
        x2 = self.gru_2_2(x2)[0]
        x2r_laststep = x2[-1]
        np.save('x2_save', x2r_laststep.cpu().detach().numpy(), allow_pickle=True)
        x3 = self.gru_2_3(x3)[0]
        x3r_laststep = x3[-1]
        np.save('x3_save', x3r_laststep.cpu().detach().numpy(), allow_pickle=True)
        x4 = self.gru_2_4(x4)[0]
        x4r_laststep = x4[-1]
        np.save('x4_save', x4r_laststep.cpu().detach().numpy(), allow_pickle=True)
        x5 = self.gru_2_5(x5)[0]
        x5r_laststep = x5[-1]
        np.save('x5_save', x5r_laststep.cpu().detach().numpy(), allow_pickle=True)
        x6 = self.gru_2_6(x6)[0]
        x6r_laststep = x6[-1]
        np.save('x6_save', x6r_laststep.cpu().detach().numpy(), allow_pickle=True)
        x7 = self.gru_2_7(x7)[0]
        x7r_laststep = x7[-1]
        np.save('x7_save', x7r_laststep.cpu().detach().numpy(), allow_pickle=True)
        x8 = self.gru_2_4(x8)[0]
        x8r_laststep = x8[-1]
        np.save('x8_save', x8r_laststep.cpu().detach().numpy(), allow_pickle=True)
        # plt.plot(self.tanh(x1r_laststep[1, :]).cpu().detach().numpy())
        # plt.plot(self.tanh(x2r_laststep[1, :]).cpu().detach().numpy())
        # plt.plot(self.tanh(x3r_laststep[1, :]).cpu().detach().numpy())
        # plt.plot(self.tanh(x4r_laststep[1, :]).cpu().detach().numpy())
        #
        # plt.show()
        # a1 = Variable(torch.ones(x1.size(1),x1.size(2)).type(dtype=torch.FloatTensor), requires_grad = True).cuda()
        # a1.backward(a1)
        # a2 = Variable(torch.ones(x2.size(1),x2.size(2)).type(dtype=torch.FloatTensor), requires_grad=True).cuda()
        # a2.backward(a2)
        # x7 = self.gru_2_2(x7)[0]
        # x = x8
        # x = self.softmax(x)
        # print(a1)
        # print(a2)
        # x = (x1 / (x1 + x2)) * x1 + (x2 / (x1 +x2) )* x2

        # 下面改变输入值，确定使用哪个方向
        # x =  x2
        # x = x.permute(1, 2, 0).contiguous()
        #换成最后的output
        # x = x1r_laststep + x3r_laststep
        # x = x.permute(0, 1).contiguous()

        x1 = x1r_laststep
        x1 = x1.permute(0,1).contiguous()

        x2 = x2r_laststep
        x2 = x2.permute(0,1).contiguous()

        x3 = x3r_laststep
        x3 = x3.permute(0,1).contiguous()

        x4 = x4r_laststep
        x4 = x4.permute(0,1).contiguous()

        x5 = x5r_laststep
        x5 = x5.permute(0, 1).contiguous()

        x6 = x6r_laststep
        x6 = x6.permute(0, 1).contiguous()

        x7 = x7r_laststep
        x7 = x7.permute(0, 1).contiguous()

        x8 = x8r_laststep
        x8 = x8.permute(0, 1).contiguous()


        # print(x.shape) #100,64,25
        # x1r = x1.permute(1, 2, 0).contiguous()
        # x2r = x2.permute(1, 2, 0).contiguous()
        # x3r = x3.permute(1, 2, 0).contiguous()
        # x4r = x4.permute(1, 2, 0).contiguous()
        # x5r = x5.permute(1, 2, 0).contiguous()
        # x6r = x6.permute(1, 2, 0).contiguous()
        # x7r = x7.permute(1, 2, 0).contiguous()
        # x8r = x8.permute(1, 2, 0).contiguous()

        # x = x.view(x.size(0), -1)

        x1 = x1.view(x1.size(0), -1)
        print('x1.view',x1.shape)
        # np.save('x1_save', x1.cpu().detach().numpy(), allow_pickle=True)
        x2 = x2.view(x2.size(0), -1)
        # np.save('x2_save', x2.cpu().detach().numpy(), allow_pickle=True)
        x3 = x3.view(x3.size(0), -1)
        # np.save('x3_save', x3.cpu().detach().numpy(), allow_pickle=True)
        x4 = x4.view(x4.size(0), -1)
        # np.save('x4_save', x4.cpu().detach().numpy(), allow_pickle=True)
        x5 = x5.view(x5.size(0), -1)
        # np.save('x5_save', x5.cpu().detach().numpy(), allow_pickle=True)
        x6 = x6.view(x6.size(0), -1)
        # np.save('x6_save', x6.cpu().detach().numpy(), allow_pickle=True)
        x7 = x7.view(x7.size(0), -1)
        # np.save('x7_save', x7.cpu().detach().numpy(), allow_pickle=True)
        x8 = x8.view(x8.size(0), -1)
        # np.save('x8_save', x8.cpu().detach().numpy(), allow_pickle=True)
        # print(x.shape) #100,1600=64*25
        # x1r = x1r.view(x1r.size(0), -1)
        # x2r = x2r.view(x2r.size(0), -1)
        # x3r = x3r.view(x3r.size(0), -1)
        # x4r = x4r.view(x4r.size(0), -1)
        # x5r = x5r.view(x5r.size(0), -1)
        # x6r = x6r.view(x6r.size(0), -1)
        # x7r = x7r.view(x7r.size(0), -1)
        # x8r = x8r.view(x8r.size(0), -1)

        # x = self.gru_bn_2(x)
        # x = self.gru_bn_laststep(x)
        x1 = self.gru_bn_laststep(x1)
        print('x1.shape',x1.shape) #[16,64]
        print('x1_save.shape', x1.shape)
        # np.save('x1_save', x1.cpu().detach().numpy(), allow_pickle=True)

        x1_b1 = x1[1,:]
        # plt.subplot(241).set_title('direction-1')
        # plt.plot(x1_b1.cpu().detach().numpy())
        # print('x1_b1.shape',x1_b1.shape)
        x1_b1 = x1_b1.unsqueeze(0)
        # print('x1_b1.shape', x1_b1.shape)

        x2 = self.gru_bn_laststep(x2)
        # np.save('x2_save', x2.cpu().detach().numpy(), allow_pickle=True)
        x2_b1 = x2[1,:]
        # plt.subplot(242).set_title('direction-2')
        # plt.plot(x2_b1.cpu().detach().numpy())
        x2_b1 = x2_b1.unsqueeze(0)

        x3 = self.gru_bn_laststep(x3)
        # np.save('x3_save', x3.cpu().detach().numpy(), allow_pickle=True)
        x3_b1 = x3[1,:]
        # plt.subplot(243).set_title('direction-3')
        # plt.plot(x3_b1.cpu().detach().numpy())
        x3_b1 = x3_b1.unsqueeze(0)

        x4 = self.gru_bn_laststep(x4)
        # np.save('x4_save', x4.cpu().detach().numpy(), allow_pickle=True)
        x4_b1 = x4[1,:]
        # plt.subplot(244).set_title('direction-4')
        # plt.plot(x4_b1.cpu().detach().numpy())
        x4_b1 = x4_b1.unsqueeze(0)

        x5 = self.gru_bn_laststep(x5)
        # np.save('x5_save', x5.cpu().detach().numpy(), allow_pickle=True)
        x5_b1 = x5[1,:]
        # plt.subplot(245).set_title('direction-5')
        # plt.plot(x5_b1.cpu().detach().numpy())
        x5_b1 = x5_b1.unsqueeze(0)

        x6 = self.gru_bn_laststep(x6)
        # np.save('x6_save', x6.cpu().detach().numpy(), allow_pickle=True)
        x6_b1 = x6[1,:]
        # plt.subplot(246).set_title('direction-6')
        # plt.plot(x6_b1.cpu().detach().numpy())
        x6_b1 = x6_b1.unsqueeze(0)

        x7 = self.gru_bn_laststep(x7)
        # np.save('x7_save', x7.cpu().detach().numpy(), allow_pickle=True)
        x7_b1 = x7[1,:]
        # plt.subplot(247).set_title('direction-7')
        # plt.plot(x7_b1.cpu().detach().numpy())
        x7_b1 = x7_b1.unsqueeze(0)

        x8 = self.gru_bn_laststep(x8)
        # np.save('x8_save', x8.cpu().detach().numpy(), allow_pickle=True)
        x8_b1 = x8[1,:]
        # plt.subplot(248).set_title('direction-8')
        # plt.plot(x8_b1.cpu().detach().numpy())
        x8_b1 = x8_b1.unsqueeze(0)
        # plt.show()

        # M = torch.cat([x1_b1,x2_b1,x3_b1,x4_b1,x5_b1,x6_b1,x7_b1,x8_b1],dim=0)
        # print('M.shape',M.shape)
        # pca = PCA(n_components=2)
        # M = M.cpu().detach().numpy()
        # reduced = pca.fit_transform(M)
        # print(reduced.shape)
        # # t = reduced
        # t = reduced.transpose()
        # print(t.shape)
        # cValue = ['b', 'c', 'g', 'k', 'm', 'r', '#FF8C00', 'y'] #blue第一,cyan第二,绿色第三,black_4,紫色_5,红色6,橘色7,黄色8
        # plt.scatter(t[0], t[1], s=100, cmap=True, c=cValue, marker='^')
        # plt.legend()
        # plt.show()
        # x1r = self.gru_bn_2(x1r)
        # x2r = self.gru_bn_2(x2r)
        # x3r = self.gru_bn_2(x3r)
        # x4r = self.gru_bn_2(x4r)
        # x5r = self.gru_bn_2(x5r)
        # x6r = self.gru_bn_2(x6r)
        # x7r = self.gru_bn_2(x7r)
        # x8r = self.gru_bn_2(x8r)

        x1_save = self.relu(x1)
        x2_save = self.relu(x2)
        x3_save = self.relu(x3)
        x4_save = self.relu(x4)
        x5_save = self.relu(x5)
        x6_save = self.relu(x6)
        x7_save = self.relu(x7)
        x8_save = self.relu(x8)

        # x1r = self.relu(x1r)
        # x2r = self.relu(x2r)
        # x3r = self.relu(x3r)
        # x4r = self.relu(x4r)
        # x5r = self.relu(x5r)
        # x6r = self.relu(x6r)
        # x7r = self.relu(x7r)
        # x8r = self.relu(x8r)

        # x = self.dropout(x)
        # x1r = self.dropout(x1r)
        # plt.subplot(3, 3, 2).set_title('direction-1')
        # plt.xlabel('Feature size', fontdict={'size': 15}, fontweight='bold')
        # plt.ylabel('Feature Values', fontdict={'size': 15}, fontweight='bold')
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.plot(x1r[0, :].cpu().detach().numpy())

        # x2r = self.dropout(x2r)
        # plt.subplot(3, 3, 3).set_title('direction-2')
        # plt.xlabel('Feature size', fontdict={'size': 15}, fontweight='bold')
        # plt.ylabel('Feature Values', fontdict={'size': 15}, fontweight='bold')
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.plot(x2r[0, :].cpu().detach().numpy())

        # x3r = self.dropout(x3r)
        # plt.subplot(3, 3, 4)
        # plt.xlabel('Feature size', fontdict={'size': 15}, fontweight='bold')
        # plt.ylabel('Feature Values', fontdict={'size': 15}, fontweight='bold')
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.plot(x3r[0, :].cpu().detach().numpy())

        # x4r = self.dropout(x4r)
        # plt.subplot(3, 3, 5)
        # plt.xlabel('Feature size', fontdict={'size': 15}, fontweight='bold')
        # plt.ylabel('Feature Values', fontdict={'size': 15}, fontweight='bold')
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.plot(x4r[0, :].cpu().detach().numpy())

        # x5r = self.dropout(x5r)
        # plt.subplot(3, 3, 6)
        # plt.xlabel('Feature size', fontdict={'size': 15}, fontweight='bold')
        # plt.ylabel('Feature Values', fontdict={'size': 15}, fontweight='bold')
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.plot(x5r[0, :].cpu().detach().numpy())

        # x6r = self.dropout(x6r)
        # plt.subplot(3, 3, 7)
        # plt.xlabel('Feature size', fontdict={'size': 15}, fontweight='bold')
        # plt.ylabel('Feature Values', fontdict={'size': 15}, fontweight='bold')
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.plot(x6r[0, :].cpu().detach().numpy())

        # x7r = self.dropout(x7r)
        # plt.subplot(3, 3, 8)
        # plt.xlabel('Feature size', fontdict={'size': 15}, fontweight='bold')
        # plt.ylabel('Feature Values', fontdict={'size': 15}, fontweight='bold')
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.plot(x7r[0, :].cpu().detach().numpy())

        # x8r = self.dropout(x8r)
        # plt.subplot(3, 3, 9)
        # plt.xlabel('Feature size', fontdict={'size': 15}, fontweight='bold')
        # plt.ylabel('Feature Values', fontdict={'size': 15}, fontweight='bold')
        # plt.xticks(fontsize=10)x1 = self.fc_2(x1)
        # plt.yticks(fontsize=10)
        # plt.plot(x8r[0, :].cpu().detach().numpy())
        # plt.show()

        # plt.subplot(3, 3, 1).set_title('Spectral signatures in a patch')
        # plt.xlabel('Band Numbers', fontdict={'size': 15}, fontweight='bold')
        # plt.ylabel('Spectral Values Values', fontdict={'size': 15}, fontweight='bold')
        # plt.plot(direction_1[0, :, :].cpu().detach().numpy())

        # x = self.fc_2(x)


        x1 = self.fc_laststep(x1)
        x1_fc_b1 = x1[1, :]
        x1_fc_b1 = x1_fc_b1.unsqueeze(0)

        x2 = self.fc_laststep(x2)
        x2_fc_b1 = x2[1, :]
        x2_fc_b1 = x2_fc_b1.unsqueeze(0)

        x3 = self.fc_laststep(x3)
        x3_fc_b1 = x3[1, :]
        x3_fc_b1 = x3_fc_b1.unsqueeze(0)

        x4 = self.fc_laststep(x4)
        x4_fc_b1 = x4[1, :]
        x4_fc_b1 = x4_fc_b1.unsqueeze(0)

        x5 = self.fc_laststep(x5)
        x5_fc_b1 = x5[1, :]
        x5_fc_b1 = x5_fc_b1.unsqueeze(0)

        x6 = self.fc_laststep(x6)
        x6_fc_b1 = x6[1, :]
        x6_fc_b1 = x6_fc_b1.unsqueeze(0)

        x7 = self.fc_laststep(x7)
        x7_fc_b1 = x7[1, :]
        x7_fc_b1 = x7_fc_b1.unsqueeze(0)

        x8 = self.fc_laststep(x8)
        x8_fc_b1 = x8[1, :]
        x8_fc_b1 = x8_fc_b1.unsqueeze(0)

        # M2 = torch.cat([x1_fc_b1, x2_fc_b1, x3_fc_b1, x4_fc_b1, x5_fc_b1, x6_fc_b1, x7_fc_b1, x8_fc_b1], dim=0)
        # print('M2.shape', M2.shape)
        # pca = PCA(n_components=2)
        # M2 = M2.cpu().detach().numpy()
        # reduced2 = pca.fit_transform(M2)
        # print(reduced2.shape)
        # # t = reduced
        # t2 = reduced2.transpose()
        # print(t2.shape)
        # cValue = ['b', 'c', 'g', 'k', 'm', 'r', '#FF8C00', 'y']  # blue第一,cyan第二,绿色第三,black_4,紫色_5,红色6,橘色7,黄色8
        # plt.scatter(t2[0], t2[1], s=100, cmap=True, c=cValue)
        # plt.show()

        # x1_fc_b1 = self.softmax(x1_fc_b1)
        # # print('x1_fc_b1',x1_fc_b1.shape)
        # plt.subplot(241).set_title('direction-1')
        # plt.plot(x1_fc_b1[0,:].cpu().detach().numpy())
        #
        # x2_fc_b1 = self.softmax(x2_fc_b1)
        # plt.subplot(242).set_title('direction-2')
        # plt.plot(x2_fc_b1[0, :].cpu().detach().numpy())
        #
        # x3_fc_b1 = self.softmax(x3_fc_b1)
        # plt.subplot(243).set_title('direction-3')
        # plt.plot(x3_fc_b1[0, :].cpu().detach().numpy())
        #
        # x4_fc_b1 = self.softmax(x4_fc_b1)
        # plt.subplot(244).set_title('direction-4')
        # plt.plot(x4_fc_b1[0, :].cpu().detach().numpy())
        #
        # x5_fc_b1 = self.softmax(x5_fc_b1)
        # plt.subplot(245).set_title('direction-5')
        # plt.plot(x5_fc_b1[0, :].cpu().detach().numpy())
        #
        # x6_fc_b1 = self.softmax(x6_fc_b1)
        # plt.subplot(246).set_title('direction-6')
        # plt.plot(x6_fc_b1[0, :].cpu().detach().numpy())
        #
        # x7_fc_b1 = self.softmax(x7_fc_b1)
        # plt.subplot(247).set_title('direction-7')
        # plt.plot(x7_fc_b1[0, :].cpu().detach().numpy())
        #
        # x8_fc_b1 = self.softmax(x8_fc_b1)
        # plt.subplot(248).set_title('direction-8')
        # plt.plot(x8_fc_b1[0, :].cpu().detach().numpy())

        # plt.show()

        # M3 = torch.cat([x1_fc_b1, x2_fc_b1, x3_fc_b1, x4_fc_b1, x5_fc_b1, x6_fc_b1, x7_fc_b1, x8_fc_b1], dim=0)
        # print('M3.shape', M3.shape)
        # pca = PCA(n_components=2)
        # M3 = M3.cpu().detach().numpy()
        # reduced3 = pca.fit_transform(M3)
        # print(reduced3.shape)
        # # t = reduced
        # t3 = reduced3.transpose()
        # print(t3.shape)
        # cValue = ['b', 'c', 'g', 'k', 'm', 'r', '#FF8C00', 'y']  # blue第一,cyan第二,绿色第三,black_4,紫色_5,红色6,橘色7,黄色8
        # plt.scatter(t3[0], t3[1], s=100, cmap=True, c=cValue, marker='v')

        # plt.show()

        x1_save = self.fc_laststep(x1_save)
        # np.save('x1_save', x1_save.cpu().detach().numpy(), allow_pickle=True)

        x2_save = self.fc_laststep(x2_save)
        # np.save('x2_save', x2_save.cpu().detach().numpy(), allow_pickle=True)

        x3_save = self.fc_laststep(x3_save)
        # np.save('x3_save', x3_save.cpu().detach().numpy(), allow_pickle=True)

        x4_save = self.fc_laststep(x4_save)
        # np.save('x4_save', x4_save.cpu().detach().numpy(),  allow_pickle=True)

        x5_save = self.fc_laststep(x5_save)
        # np.save('x5_save', x5_save.cpu().detach().numpy(),  allow_pickle=True)

        x6_save = self.fc_laststep(x6_save)
        # np.save('x6_save', x6_save.cpu().detach().numpy(),  allow_pickle=True)

        x7_save = self.fc_laststep(x7_save)
        # np.save('x7_save', x7_save.cpu().detach().numpy(),allow_pickle=True)

        x8_save = self.fc_laststep(x8_save)
        # np.save('x8_save', x8_save.cpu().detach().numpy(),  allow_pickle=True)
        # x2r = self.fc_3(x2r)
        # x3r = self.fc_3(x3r)
        # x4r = self.fc_3(x4r)
        # x5r = self.fc_3(x5r)
        # x6r = self.fc_3(x6r)
        # x7r = self.fc_3(x7r)
        # x8r = self.fc_3(x8r)
        x = x1_save + x2_save + x3_save +x4_save +x5_save + x6_save + x7_save +x8_save

        # x1r = self.softmax(x1r)
        # # print('x1r', x1r[0, :])
        # plt.subplot(332).set_title('Probability for each class')
        # plt.plot(x1r[0, :].cpu().detach().numpy())
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.xlabel('Class numbers', fontdict={'size': 15}, fontweight='bold')
        # plt.ylabel('Probability', fontdict={'size': 15}, fontweight='bold')
        #
        # x2r = self.softmax(x2r)
        # # print('x2r', x2r[0, :])
        # plt.subplot(333).set_title('Probability for each class')
        # plt.plot(x2r[0, :].cpu().detach().numpy())
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.xlabel('Class numbers', fontdict={'size': 15}, fontweight='bold')
        # plt.ylabel('Probability', fontdict={'size': 15}, fontweight='bold')
        #
        # x3r = self.softmax(x3r)
        # plt.subplot(334)
        # plt.plot(x3r[0, :].cpu().detach().numpy())
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.xlabel('Class numbers', fontdict={'size': 15}, fontweight='bold')
        # plt.ylabel('Probability', fontdict={'size': 15}, fontweight='bold')
        #
        # x4r = self.softmax(x4r)
        # plt.subplot(335)
        # plt.plot(x4r[0, :].cpu().detach().numpy())
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.xlabel('Class numbers', fontdict={'size': 15}, fontweight='bold')
        # plt.ylabel('Probability', fontdict={'size': 15}, fontweight='bold')
        #
        # x5r = self.softmax(x5r)
        # plt.subplot(336)
        # plt.plot(x5r[0, :].cpu().detach().numpy())
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.xlabel('Class numbers', fontdict={'size': 15}, fontweight='bold')
        # plt.ylabel('Probability', fontdict={'size': 15}, fontweight='bold')
        #
        # x6r = self.softmax(x6r)
        # plt.subplot(337)
        # plt.plot(x6r[0, :].cpu().detach().numpy())
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.xlabel('Class numbers', fontdict={'size': 15}, fontweight='bold')
        # plt.ylabel('Probability', fontdict={'size': 15}, fontweight='bold')
        #
        # x7r = self.softmax(x7r)
        # plt.subplot(338)
        # plt.plot(x7r[0, :].cpu().detach().numpy())
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.xlabel('Class numbers', fontdict={'size': 15}, fontweight='bold')
        # plt.ylabel('Probability', fontdict={'size': 15}, fontweight='bold')
        #
        # x8r = self.softmax(x8r)
        # plt.subplot(339)
        # plt.plot(x8r[0, :].cpu().detach().numpy())
        # plt.xlabel('Class numbers', fontdict={'size': 15}, fontweight='bold')
        # plt.ylabel('Probability', fontdict={'size': 15}, fontweight='bold')
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.show()
        return x

class zhouConstraintRNN(nn.Module):
    """
    one direction rnn with spatial consideration which has a patch size
    """
    @staticmethod
    def weight_init(m):
        # All weight matrices in our RNN and bias vectors are initialized with a uniform distribution, and the values of these weight matrices and bias vectors are initialized in the range [−0.1,0.1]
        if isinstance(m, (nn.Linear, nn.RNN, nn.Conv3d)):
            init.kaiming_uniform_(m.weight.data)
            init.kaiming_uniform_(m.bias.data)

    def __init__(self, input_channels, n_classes, patch_size=5):
        # The proposed network model uses a single recurrent layer that adopts our modified GRUs of size 64 with sigmoid gate activation and PRetanh activation functions for hidden representations
        super(zhouConstraintRNN, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.gru = nn.RNN(patch_size**2, patch_size**2, 1, bidirectional=False, batch_first=False)  # TODO: try to change this ?
        self.gru_bn = nn.BatchNorm1d(patch_size**2 * input_channels)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(patch_size**2 * input_channels, n_classes)
        # self.conv_3x3 = nn.Conv3d(
        #     1, 103, (input_channels, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        # self.conv_1x1 = nn.Conv3d(
        #     1, 103, (input_channels, 1, 1), stride=(1, 1, 1), padding=0)
        # self.conv_5_5 = nn.Conv3d(
        #     1, 103, (input_channels, 5, 5), stride=(1, 1, 1), padding=(0, 2, 2)
        # )


    def forward(self, x):  # 初始是第1方向
        # print('input_size',x.shape)
        x = x.squeeze(1)
        x1 =x
        # print('x1_size',x1.shape) #[100,103,5,5]
        x1r = x1.reshape(x1.shape[0], x1.shape[1], -1)
        plt.subplot(121)
        plt.plot(x1r[0, :, :].cpu().detach().numpy())
        # print('x1r_shape',x1r.shape) #[100,103,25]

        for i in range (x1r.shape[0]):
            x_batch_i = x1r[i,:,:]
            # print('x_batch_i',x_batch_i.shape)
            vectors = []
            for s in range(x_batch_i.shape[1]):
                vector = x_batch_i[:, s]
                vectors.append(vector)  # 得到vector
            # print('vectors', len(vectors))
            # print('vectors', vectors)
            xc = vectors[int((len(vectors) - 1) / 2)]  # 中心vector
            # print('xc', xc)
            distances = []  # 各个vector与中心vector的距离
            SAM = []
            distances_power2 = []  # 距离的平方
            sum_distances = 0
            for n in range(len(vectors)):
                distance = torch.dist(xc, vectors[n], p=2)
                sum_distances = sum_distances + distance  # 距离之和
                distance_p_2 = distance ** 2
                distances.append(distance)
                distances_power2.append(distance_p_2)
            # print('distances', distances)
            # print('distance_power2', distances_power2)
            # print('sum_distance', sum_distances)
            sum_distances_p_2 = sum_distances ** 2  # 距离之和的平方
            # print('rou_p_2', sum_distances_p_2)
            # print('len(dis_p2)', len(distances_power2))

            Hs = []  # 得到权重
            for k in range(len(distances_power2)):
                h = -(distances_power2[k] / (2 * sum_distances_p_2))
                H = torch.exp(h)
                Hs.append(H)

            # print('Hs', Hs)

            # print('len(Hs)', len(Hs))
            # print('len(vectors)', len(vectors))
            New_Vectors = []
            for j in range(len(Hs)) and range(len(vectors)):
                New_Vector = Hs[j] * vectors[j]
                New_Vectors.append(New_Vector)
            # print('New_Vectors', New_Vectors)  # 加权之后的新Vector

            np_New_Vectors = []
            for m in range(len(New_Vectors)):
                np_New_Vector = New_Vectors[m].cpu().numpy().copy()
                np_New_Vectors.append(np_New_Vector)
            # print('np_New_Vectors', np_New_Vectors)

            M_np = np.array(np_New_Vectors).T
            print('M_np', M_np.shape)

            M_torch_s = torch.from_numpy(M_np).cuda()
            # print('M_torch', M_torch)
            # print('M_torch_s', M_torch_s.shape)
            if i == 0:
                M_torch = M_torch_s.unsqueeze(dim=0)
                print('M_torch_s_un', M_torch.shape)
            else:
                M_torch = torch.cat([M_torch, M_torch_s.unsqueeze(dim=0)], dim=0)
                # print('M_torch', M_torch.shape)


        print('M_torch',M_torch.shape)
        x1r = M_torch
        print('x1r',x1r.shape)
        plt.subplot(122)
        plt.plot(M_torch[0,:,:].cpu().detach().numpy())
        plt.show()
        x = x1r.permute(1, 0, 2)
        # print("x",x.shape)

        # # 第二方向
        # # x2 = x1r.cpu()
        # # x2rn = np.flip(x2.numpy(), axis=2).copy()
        # # x2rt = torch.from_numpy(x2rn)
        # # x2r = x2rt.cuda()
        # # x = x2r.permute(1, 0, 2)
        # #3
        # # x3 = torch.transpose(x1, 2, 3)
        # # x3r = x3.reshape(x3.shape[0], x3.shape[1], -1)
        # # # x = x3r.permute(1, 0, 2)
        # # #4
        # # x4 = x3r.cpu()
        # # x4rn = np.flip(x4.numpy(), axis=2).copy()
        # # x4rt = torch.from_numpy(x4rn)
        # # x4r = x4rt.cuda()
        # # x = x4r.permute(1, 0, 2)
        #
        # x5 = torch.rot90(x1, 1, (2, 3))
        # x5r = x5.reshape(x5.shape[0], x5.shape[1], -1)
        # x = x5r.permute(1, 0, 2)
        #
        # x6 = x5r.cpu()
        # x6rn = np.flip(x6.numpy(), axis=2).copy()
        # x6rt = torch.from_numpy(x6rn)
        # x6r = x6rt.cuda()
        # x = x6r.permute(1, 0, 2)
        # #
        # x7 = torch.transpose(x5, 2, 3)
        # x7r = x7.reshape(x7.shape[0], x7.shape[1], -1)
        # # x = x7r.permute(1, 0, 2)
        # # #
        # x8 = x7r.cpu()
        # x8rn = np.flip(x8.numpy(), axis=2).copy()
        # x8rt = torch.from_numpy(x8rn)
        # x8r = x8rt.cuda()
        # x = x8r.permute(1, 0, 2)

        x = x
        # 导入RNN
        # print('x',x.shape)
        x = self.gru(x)[0]
        # print('4-1', x.shape)
        x = x.permute(1, 2, 0).contiguous()
        x = x.view(x.size(0), -1)
        # print('5', x.shape)
        x = self.gru_bn(x)
        # x = self.sigmoid(x)
        x = self.tanh(x)
        # x = self.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class zhouRNNC(nn.Module):
    """
    one direction rnn with spatial consideration which has a patch size
    """
    @staticmethod
    def weight_init(m):
        # All weight matrices in our RNN and bias vectors are initialized with a uniform distribution, and the values of these weight matrices and bias vectors are initialized in the range [−0.1,0.1]
        if isinstance(m, (nn.Linear, nn.GRU)):
            init.uniform(m.weight.data, -0.1, 0.1)
            init.uniform(m.bias.data, -0.1, 0.1)

    def __init__(self, input_channels, n_classes, patch_size=5):
        # The proposed network model uses a single recurrent layer that adopts our modified GRUs of size 64 with sigmoid gate activation and PRetanh activation functions for hidden representations
        super(zhouRNNC, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.gru = nn.GRU(patch_size**2, 64, 2, bidirectional=False, batch_first=False)  # TODO: try to change this ?
        self.gru_bn = nn.BatchNorm1d(64 * input_channels)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(64 * input_channels, n_classes)
        self.addw = self.add_weight

    def forward(self, x):  # 初始是第三方向
        print('0', x.shape)
        x3 = x.squeeze(1)
        print('1', x3.shape)
        # 生成第一方向
        #x1 = torch.transpose(x3, 2, 3)
        # # 生成第二方向
        # x2 = x1.cpu()
        # x2n = np.flip(x2.numpy(),axis=2).copy()
        # print('3-1',x2n.shape)
        # x2t = torch.from_numpy(x2n)
        # print('3-2',x2t.shape)
        # x3 = x2t.cuda() #把值给x
        x = x3
        x = x.reshape(x.shape[0],103,-1)
        print('1-1',x.shape)
        # x = self.addw(x)
        print('2',x.shape)
        x = x.permute(1, 0, 2)
        print('3', x.shape)
        x = x
        # #生成第四方向 从第三方向来
        # x4 = x3.cpu()
        # x4n = np.flip(x4.numpy(),axis=2).copy()
        # print('3-1',x4n.shape)
        # x4t = torch.from_numpy(x4n)
        # print('3-2',x4t.shape)
        # x = x4t.cuda() #把值给x
        #导入RNN
        x = self.gru(x)[0]
        print('4-1',x.shape)
        x = x.permute(1, 2, 0).contiguous()
        x = x.view(x.size(0),-1)
        print('5',x.shape)
        x = self.gru_bn(x)
        print('6',x.shape)
        x = self.tanh(x)
        print('7',x.shape)
        x = self.fc(x)
        return x

class zhouDREtAl(nn.Module):

    #3-D DRConv NO PCA, NO POOLING , NO RESHAPE, NO DROPOUT, USE ALL THE INFORMATION
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.xavier_uniform_(m.weight.data)
            init.constant_(m.bias.data,0)

    def __init__(self, input_channels, n_classes, n_planes=16, patch_size=9):
        super(zhouDREtAl, self).__init__()
        self.input_channels = input_channels
        self.n_planes = n_planes
        self.patch_size = patch_size
        # DR Conv(my)
        self.convDR = nn.Conv3d(1, n_planes, (34, 3, 3), padding=(0, 0, 0), stride=(34, 1, 1))
        self.conv2d1 = nn.Conv2d(in_channels=input_channels,out_channels=n_planes,kernel_size=3,stride=3)
        # ouput of DRConv into 2rd Conv.
        self.conv1 = nn.Conv3d(n_planes, n_planes, (3, 1, 1), padding=(1, 0, 0), stride=(1, 1, 1))
        # the number of kernels in the second convolution layer is set to be
        # twice as many as that in the first convolution layer
        self.conv2d2 = nn.Conv2d(in_channels=n_planes,out_channels=n_planes*2,kernel_size=3)
        self.conv2 = nn.Conv3d(n_planes, 2 * n_planes, (3, 3, 3), padding=(0, 0, 0), stride=(1, 1, 1))
        # self.dropout = nn.Dropout(p=0.5)
        self.features_size = self._get_final_flattened_size()
        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.convDR(x)
            x = self.conv1(x)
            x = self.conv2(x)
            _, t, c, w, h = x.size()
            print(t, c, w, h)
        return t * c * w * h

    def forward(self, x):
        # print(x.shape) #100,1,200,9,9
        x = F.relu(self.convDR(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.features_size)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def train(net, optimizer, criterion, data_loader, epoch, scheduler=None,
          display_iter=100, device=torch.device('cpu'), display=viz_tsne,
          val_loader=True, supervision='full'):
    #device=torch.device('cuda:0)
    global target

    """
    Training loop to optimize a network for several epochs and a specified loss

    Args:
        net: a PyTorch model
        optimizer: a PyTorch optimizer
        data_loader: a PyTorch dataset loader
        epoch: int specifying the number of training epochs
        criterion: a PyTorch-compatible loss function, e.g. nn.CrossEntropyLoss
        device (optional): torch device to use (defaults to CPU)
        display_iter (optional): number of iterations before refreshing the
        display (False/None to switch off).
        scheduler (optional): PyTorch scheduler
        val_loader (optional): validation dataset
        supervision (optional): 'full' or 'semi'
    """
    net.to(device)

    if criterion is None:
        raise Exception("Missing criterion. You must specify a loss function.")

    save_epoch = epoch // 20 if epoch > 20 else 1
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    iter_ = 1
    loss_win, val_win = None, None
    val_accuracies = []

    for e in tqdm(range(1, epoch + 1), desc="Training the network", colour='red'):
        # Set the network to training mode
        net.train()
        avg_loss = 0.
        psnr_sum = 0
        num_psnr = 0
        ssim_sum = 0.0
        num_pairs = 0
        # Run the training loop for one epoch
        for batch_idx, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader), colour='blue'):
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            if val_loader is not None:
                # val_acc = val(net, val_loader, device=torch.device('cuda:0'), supervision=supervision)
                val_acc = val(net, val_loader, device=device, supervision=supervision)
                val_accuracies.append(val_acc)
                metric = -val_acc

            if supervision == 'full':
                # output = net(data)
                outs = net(data)
                output, rec = outs
                # loss = criterion(output, target)

                loss = criterion[0](output, target) + net.aux_loss_weight * criterion[1](rec, data)

                if e % 40 == 0:
                    # viz.matplot(plt)
                    viz.line(data[1, :, :, 9 // 2, 9 // 2].squeeze().cpu().detach().numpy(), opts={'title': 'data' + str(e) + str(target[1])})
                    viz.line(rec[1,:].cpu().detach().numpy(), opts={'title': 'regrss' + str(e) + str(target[1])})

                # use psnr function:
                psnr = calculate_psnr(data[:, :, :, 9 // 2, 9 // 2].squeeze().cpu().detach().numpy(), rec.detach().cpu().numpy())
                psnr_sum += psnr
                num_psnr += 1

                # l1_lambda = 0.001
                # l1_norm = sum(p.abs().sum() for p in net.parameters())
                # loss = loss + l1_lambda * l1_norm

            elif supervision == 'semi':
                outs = net(data)
                output, rec = outs
                loss = criterion[0](output, target) + net.aux_loss_weight * criterion[1](rec, data)
            else:
                raise ValueError("supervision mode \"{}\" is unknown.".format(supervision))

            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            losses[iter_] = loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_ + 1])

            if display_iter and iter_ % display_iter == 0:
                string = 'Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tVal_ACC: {:.5f}'
                string = string.format(
                    e, epoch, batch_idx *
                    len(data), len(data) * len(data_loader),
                    100. * batch_idx / len(data_loader), mean_losses[iter_], val_acc)
                update = None if loss_win is None else 'append'
                loss_win = display.line(
                    X=np.arange(iter_ - display_iter, iter_),
                    Y=mean_losses[iter_ - display_iter:iter_],
                    win=loss_win,
                    update=update,
                    opts={'title': "Training loss",
                          'xlabel': "Iterations",
                          'ylabel': "Loss"
                         }
                )
                tqdm.write(f"\033[1m{string}\033[0m")

                if len(val_accuracies) > 0:
                    val_win = display.line(Y=np.array(val_accuracies),
                                           X=np.arange(len(val_accuracies)),
                                           win=val_win,
                                           opts={'title': "Validation accuracy",
                                                 'xlabel': "Epochs",
                                                 'ylabel': "Accuracy"
                                                })
            iter_ += 1
            del(data, target, loss, output)

        avg_loss /= len(data_loader)

        # After each epoch, compute the average PSNR
        avg_psnr = psnr_sum / num_psnr
        print(f"\033[91m Average PSNR for epoch {e}: {avg_psnr}\033[0m")


    outputs = []
    labels = []
    # Add t-SNE visualization here
    if e % 10 == 0:  # for instance, set display_epoch = 10 to display every 10 epochs
        with torch.no_grad():
            # Get the outputs and labels for the entire dataset
            for data, target in data_loader:
                data = data.to(device)
                output, _ = net(data)  # assuming your network returns the output and something else
                outputs.append(output)
                labels.append(target)
        # Concatenate all outputs and labels
        outputs = torch.cat(outputs, dim=0)
        labels = torch.cat(labels, dim=0)
        # Convert the outputs to 2D for t-SNE
        outputs_reshaped = outputs.view(outputs.shape[0], -1).cpu().numpy()
        labels = labels.cpu().numpy()
        # Run t-SNE
        outputs_embedded = TSNE(n_components=2, verbose=True).fit_transform(outputs_reshaped)
        # Visualize the result with different colors for each class
        num_classes = labels.max() + 1
        plt.figure(figsize=(10, 10))
        for i in range(num_classes):
            indices = labels == i
            plt.scatter(outputs_embedded[indices, 0], outputs_embedded[indices, 1], label=f'Class {i}',linewidths=1, ec="black")
        # plt.title(f't-SNE visualization at epoch {e}', fontsize = 20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('t-SNE component 1', fontsize=20)
        plt.ylabel('t-SNE component 2', fontsize=20)
        plt.legend(fontsize='x-large')
        # plt.show()
        display.matplot(plt)

        if val_loader is not None:
            val_acc = val(net, val_loader, device=device, supervision=supervision)
            val_accuracies.append(val_acc)
            metric = -val_acc
        else:
            metric = avg_loss

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric)
        elif scheduler is not None:
            scheduler.step()

        # Save the weights
        if e % save_epoch == 0:
            save_model(net, camel_to_snake(str(net.__class__.__name__)), data_loader.dataset.name, epoch=e, metric=abs(metric))

def save_model(model, model_name, dataset_name, **kwargs):
     model_dir = './checkpoints/' + model_name + "/" + dataset_name + "/"
     if not os.path.isdir(model_dir):
         os.makedirs(model_dir, exist_ok=True)
     if isinstance(model, torch.nn.Module):
         filename = str(datetime.datetime.now()) + "_epoch{epoch}_{metric:.2f}".format(**kwargs)
         tqdm.write("Saving neural network weights in {}".format(filename))
         torch.save(model.state_dict(), model_dir + filename + '.pth')
     else:
         filename = str(datetime.datetime.now())
         tqdm.write("Saving model params in {}".format(filename))
         joblib.dump(model, model_dir + filename + '.pkl')


def test(net, img, hyperparams):
    """
    Test a model on a specific image
    """
    net.eval()
    patch_size = hyperparams['patch_size']
    center_pixel = hyperparams['center_pixel']
    batch_size, device = hyperparams['batch_size'], hyperparams['device']
    n_classes = hyperparams['n_classes']

    kwargs = {'step': hyperparams['test_stride'], 'window_size': (patch_size, patch_size)}
    probs = np.zeros(img.shape[:2] + (n_classes,))
    output_list = []
    iterations = count_sliding_window(img, **kwargs) // batch_size
    for batch in tqdm(grouper(batch_size, sliding_window(img, **kwargs)),
                      total=(iterations),
                      desc="Inference on the image",
                      colour= 'green'
                      ):
        with torch.no_grad():
            if patch_size == 1:
                data = [b[0][0, 0] for b in batch]
                data = np.copy(data)
                data = torch.from_numpy(data)
            else:
                data = [b[0] for b in batch]
                data = np.copy(data)
                data = data.transpose(0, 3, 1, 2)
                data = torch.from_numpy(data)
                data = data.unsqueeze(1)

            indices = [b[1:] for b in batch]
            data = data.to(device)
            output = net(data)
            if isinstance(output, tuple):
                output = output[0]
            output = output.to('cpu')

            if patch_size == 1 or center_pixel:
                output = output.numpy()
            else:
                output = np.transpose(output.numpy(), (0, 2, 3, 1))

            output_list.append(output)

            for (x, y, w, h), out in zip(indices, output):
                if center_pixel:
                    probs[x + w // 2, y + h // 2] += out
                else:
                    probs[x:x + w, y:y + h] += out

    return probs, output_list

#
# def plot_tsne(outputs, labels):
#     # Flatten the outputs and convert to numpy array
#     outputs = torch.cat(outputs, dim=0)
#     # outputs_flat = np.array([output.flatten() for output in outputs])
#     outputs_flat = outputs.view(outputs.shape[0], -1).cpu().numpy()
#
#     # Perform t-SNE
#     tsne = TSNE(n_components=2, verbose=True)
#     tsne_results = tsne.fit_transform(outputs_flat)
#
#     # Assuming labels are integers, get unique labels
#     unique_labels = np.unique(labels)
#
#     # Choose a color for each label
#     colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))
#
#     # Plot the results
#     for i, label in enumerate(unique_labels):
#         indices = np.where(labels == label)
#         plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], color=colors[i], label=label)
#
#     # plt.title(f't-SNE visualization at epoch {e}', fontsize = 20)
#     plt.xticks(fontsize=20)
#     plt.yticks(fontsize=20)
#     plt.xlabel('t-SNE component 1', fontsize=20)
#     plt.ylabel('t-SNE component 2', fontsize=20)
#     plt.legend(fontsize='x-large')
#     plt.show()

def val(net, data_loader, device='cpu', supervision='full'): #device='cuda'
# TODO : fix me using metrics()
    accuracy, total = 0., 0.
    ignored_labels = data_loader.dataset.ignored_labels
    for batch_idx, (data, target) in enumerate(data_loader):
        # print('data',data.shape)
        with torch.no_grad():
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)
            if supervision == 'full':
                output, rec = net(data)
            elif supervision == 'semi':
                output, rec = net(data)
            _, output = torch.max(output, dim=1)
            for out, pred in zip(output.view(-1), target.view(-1)):
                if out.item() in ignored_labels:
                    continue
                else:
                    accuracy += out.item() == pred.item()
                    total += 1
    return accuracy / total