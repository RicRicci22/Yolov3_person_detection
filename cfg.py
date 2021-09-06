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
Cfg.batch = 2
Cfg.subdivisions = 1
Cfg.width = 608
Cfg.height = 608
Cfg.channels = 3
Cfg.learning_rate = 0.001
Cfg.momentum = 0.949
Cfg.decay = 0.0005
Cfg.burn_in = 1000
Cfg.max_batches = 500500
Cfg.steps = [400000, 450000]
Cfg.policy = Cfg.steps
Cfg.scales = .1, .1
Cfg.boxes = 150  # box num
Cfg.TRAIN_EPOCHS = 10
Cfg.train_label = os.path.join(_BASE_DIR, 'datasets','visdrone','train','_annotations.txt')
Cfg.val_label = os.path.join(_BASE_DIR, 'datasets','visdrone','val','_annotations.txt')
Cfg.train_dataset_dir = os.path.join(_BASE_DIR, 'datasets','visdrone','train')
Cfg.val_dataset_dir = os.path.join(_BASE_DIR, 'datasets','visdrone','val')
Cfg.TRAIN_OPTIMIZER = 'adam'

# Data augmentation
Cfg.saturation = 1.5
Cfg.exposure = 1.5
Cfg.hue = .1
Cfg.mosaic = 1
Cfg.mixup_images = 0
Cfg.classes = 1
Cfg.track = 0
Cfg.flip = True
Cfg.flip_value = -1 # 0 - x-axis, 1 - y-axis, -1 - both axes (x & y)
Cfg.blur = 4
Cfg.gaussian_noise = True 

# Saving variables 
Cfg.dataset_name = 'visdrone'
Cfg.savings_path = os.path.join(_BASE_DIR, 'trained_weights')


if Cfg.mosaic:
    Cfg.mixup = 3
else:
    Cfg.mixup = 0 

if(Cfg.mixup_images):
    Cfg.mixup = 1

Cfg.checkpoints = os.path.join(_BASE_DIR, 'checkpoints')
Cfg.TRAIN_TENSORBOARD_DIR = os.path.join(_BASE_DIR, 'log')

Cfg.iou_type = 'iou'  # 'giou', 'diou', 'ciou'

Cfg.keep_checkpoint_max = 10
