# -*- coding: utf-8 -*-
'''
@Time          : 2020/05/06 21:05
@Author        : Tianxiaomo
@File          : Cfg.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
import os
from easydict import EasyDict


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()

Cfg.gpu = 0

# Training parameters
Cfg.classes = 1
Cfg.batch = 2
Cfg.subdivisions = 1
Cfg.width = 608
Cfg.height = 608
Cfg.channels = 3
Cfg.learning_rate = 0.0005
Cfg.momentum = 0.949
Cfg.decay = 0.0005
Cfg.burn_in = 1000
Cfg.max_batches = 500500
Cfg.steps = [400000, 450000]
Cfg.policy = Cfg.steps
Cfg.scales = .1, .1
Cfg.boxes = 10  # box num
Cfg.TRAIN_EPOCHS = 1
Cfg.train_label = os.path.join(_BASE_DIR, 'datasets','custom','train','_annotations.txt')
Cfg.val_label = os.path.join(_BASE_DIR, 'datasets','custom','val','_annotations.txt')
Cfg.train_dataset_dir = os.path.join(_BASE_DIR, 'datasets','custom','train')
Cfg.val_dataset_dir = os.path.join(_BASE_DIR, 'datasets','custom','val')
Cfg.TRAIN_OPTIMIZER = 'adam'

Cfg.custom_val_dir = os.path.join(_BASE_DIR, 'datasets','custom','val')
Cfg.custom_val_label = os.path.join(_BASE_DIR, 'datasets','custom','val','_annotations.txt')

# Data augmentation
Cfg.saturation = True
Cfg.exposure = True
Cfg.hue = True
Cfg.mosaic = False
Cfg.mixup = False
Cfg.flip = True
Cfg.gaussian_noise = True
Cfg.blur = False
Cfg.crop = True 
Cfg.rotate = True


# Saving variables 
Cfg.dataset_name = 'custom'
Cfg.savings_path = os.path.join(_BASE_DIR, 'trained_weights')


Cfg.checkpoints = os.path.join(_BASE_DIR, 'checkpoints')
Cfg.TRAIN_TENSORBOARD_DIR = os.path.join(_BASE_DIR, 'log')

Cfg.iou_type = 'iou'  # 'giou', 'diou', 'ciou'

Cfg.keep_checkpoint_max = 10
