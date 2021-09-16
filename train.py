# -*- coding: utf-8 -*-
'''
@Time          : 2020/05/06 15:07
@Author        : Tianxiaomo
@File          : train.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
import logging
import os
from collections import deque
from cv2 import sort
from torch.nn import parameter

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.nn import functional as F
from easydict import EasyDict as edict
from tool.utils import *
from dataset import Yolo_dataset
from cfg import Cfg
from models import Yolov4
from detection import Detector
from metrics import Metric
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches



# def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
#     """Calculate the Intersection of Unions (IoUs) between bounding boxes.
#     IoU is calculated as a ratio of area of the intersection
#     and area of the union.
#     Args:
#         bbox_a (array): An array whose shape is :math:`(N, 4)`.
#             :math:`N` is the number of bounding boxes.
#             The dtype should be :obj:`numpy.float32`.
#         bbox_b (array): An array similar to :obj:`bbox_a`,
#             whose shape is :math:`(K, 4)`.
#             The dtype should be :obj:`numpy.float32`.
#     Returns:
#         array:
#         An array whose shape is :math:`(N, K)`. \
#         An element at index :math:`(n, k)` contains IoUs between \
#         :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
#         box in :obj:`bbox_b`.
#     from: https://github.com/chainer/chainercv
#     """
#     if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
#         raise IndexError

#     # top left
#     if xyxy:
#         tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
#         # bottom right
#         br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
#         area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
#         area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
#     else:
#         tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
#                        (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
#         # bottom right
#         br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
#                        (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

#         area_a = torch.prod(bboxes_a[:, 2:], 1)
#         area_b = torch.prod(bboxes_b[:, 2:], 1)
#     en = (tl < br).type(tl.type()).prod(dim=2)
#     area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    
#     return area_i / (area_a[:, None] + area_b - area_i)

def bboxes_iou(bboxes_a, bboxes_b, xyxy=True, GIoU=False, DIoU=False, CIoU=False):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.
    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.
    from: https://github.com/chainer/chainercv
    https://github.com/ultralytics/yolov3/blob/eca5b9c1d36e4f73bf2f94e141d864f1c2739e23/utils/utils.py#L262-L282
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        # intersection top left
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # intersection bottom right
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min(bboxes_a[:, None, :2], bboxes_b[:, :2])
        con_br = torch.max(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # centerpoint distance squared
        rho2 = ((bboxes_a[:, None, 0] + bboxes_a[:, None, 2]) - (bboxes_b[:, 0] + bboxes_b[:, 2])) ** 2 / 4 + (
                (bboxes_a[:, None, 1] + bboxes_a[:, None, 3]) - (bboxes_b[:, 1] + bboxes_b[:, 3])) ** 2 / 4

        w1 = bboxes_a[:, 2] - bboxes_a[:, 0]
        h1 = bboxes_a[:, 3] - bboxes_a[:, 1]
        w2 = bboxes_b[:, 2] - bboxes_b[:, 0]
        h2 = bboxes_b[:, 3] - bboxes_b[:, 1]

        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        # intersection top left
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # intersection bottom right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                           (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        con_br = torch.max((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                           (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))
        # centerpoint distance squared
        rho2 = ((bboxes_a[:, None, :2] - bboxes_b[:, :2]) ** 2 / 4).sum(dim=-1)

        w1 = bboxes_a[:, 2]
        h1 = bboxes_a[:, 3]
        w2 = bboxes_b[:, 2]
        h2 = bboxes_b[:, 3]

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    area_u = area_a[:, None] + area_b - area_i
    iou = area_i / area_u

    if GIoU or DIoU or CIoU:
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            area_c = torch.prod(con_br - con_tl, 2)  # convex area
            return iou - (area_c - area_u) / area_c  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = torch.pow(con_br - con_tl, 2).sum(dim=2) + 1e-16
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w1 / h1).unsqueeze(1) - torch.atan(w2 / h2), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
    return iou


class Yolo_loss(nn.Module):
    def __init__(self, n_classes=80, n_anchors=3, device=None, batch=2):
        super(Yolo_loss, self).__init__()
        self.device = device
        self.strides = [8, 16, 32]
        image_size = 608
        self.n_classes = n_classes
        self.n_anchors = n_anchors

        #self.anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]] # ORIGINAL ANCHORS
        #self.anch_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.anch_masks= [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        #self.anchors= [[7, 11], [12, 21], [16, 32], [22, 35], [33, 48], [36, 67], [48, 77], [79, 100], [94, 168]] # VISDRONE ANCHORS
        #self.anchors = [[21,34], [29,39], [49,60], [52,61], [71,78], [91,110], [97,125], [151,160], [290,173]] # SARD ANCHORS
        self.anchors = [[14,22], [25,50], [39,57], [51,75], [52,76], [63,105], [72,125], [112,268], [135,306]] # CUSTOM DATASET ANCHORS

        self.ignore_thre = 0.5

        self.masked_anchors, self.ref_anchors, self.grid_x, self.grid_y, self.anchor_w, self.anchor_h = [], [], [], [], [], []

        for i in range(3):
            all_anchors_grid = [(w / self.strides[i], h / self.strides[i]) for w, h in self.anchors]
            #print(all_anchors_grid)
            masked_anchors = np.array([all_anchors_grid[j] for j in self.anch_masks[i]], dtype=np.float32) # shape 
            #print(masked_anchors)
            ref_anchors = np.zeros((len(all_anchors_grid), 4), dtype=np.float32)
            ref_anchors[:, 2:] = np.array(all_anchors_grid, dtype=np.float32)
            #print(ref_anchors)
            #print(ref_anchors)
            ref_anchors = torch.from_numpy(ref_anchors)
            # calculate pred - xywh obj cls
            fsize = image_size // self.strides[i]
            grid_x = torch.arange(fsize, dtype=torch.float).repeat(batch, 3, fsize, 1).to(device)
            grid_y = torch.arange(fsize, dtype=torch.float).repeat(batch, 3, fsize, 1).permute(0, 1, 3, 2).to(device)
            anchor_w = torch.from_numpy(masked_anchors[:, 0]).repeat(batch, fsize, fsize, 1).permute(0, 3, 1, 2).to(
                device)

            anchor_h = torch.from_numpy(masked_anchors[:, 1]).repeat(batch, fsize, fsize, 1).permute(0, 3, 1, 2).to(
                device)

            self.masked_anchors.append(masked_anchors)
            self.ref_anchors.append(ref_anchors)
            self.grid_x.append(grid_x)
            self.grid_y.append(grid_y)
            self.anchor_w.append(anchor_w)
            self.anchor_h.append(anchor_h)

    def build_target(self, pred, labels, batchsize, fsize, n_ch, output_id):
        # target assignment
        tgt_mask = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 4 + self.n_classes).to(device=self.device)
        obj_mask = torch.ones(batchsize, self.n_anchors, fsize, fsize).to(device=self.device) 

        # why ones????
        tgt_scale = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 2).to(self.device)
        target = torch.zeros(batchsize, self.n_anchors, fsize, fsize, n_ch).to(self.device)

        # labels = labels.cpu().data
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects different from zero!
        
        
        # Trova il centro x e y 
        truth_x_all = (labels[:, :, 2] + labels[:, :, 0]) / (self.strides[output_id] * 2) # shape batch,n_box
        truth_y_all = (labels[:, :, 3] + labels[:, :, 1]) / (self.strides[output_id] * 2)
        truth_w_all = (labels[:, :, 2] - labels[:, :, 0]) / self.strides[output_id]
        truth_h_all = (labels[:, :, 3] - labels[:, :, 1]) / self.strides[output_id]
        truth_i_all = truth_x_all.to(torch.int16).cpu().numpy()# To change? 
        truth_j_all = truth_y_all.to(torch.int16).cpu().numpy()

        for b in range(batchsize):
            n = int(nlabel[b])
            if n == 0:
                continue
            truth_box = torch.zeros(n, 4).to(self.device)
            truth_box[:n, 2] = truth_w_all[b, :n]
            truth_box[:n, 3] = truth_h_all[b, :n]
            #print(truth_box)
            truth_i = truth_i_all[b, :n]
            truth_j = truth_j_all[b, :n]
            #print(truth_i)

            # calculate iou between truth and reference anchors
            #print(truth_box)
            #print(self.ref_anchors[output_id])
            anchor_ious_all = bboxes_iou(truth_box.cpu(), self.ref_anchors[output_id], CIoU=True) # shape n_truth_box, n_anchor_boxes
            #print(anchor_ious_all)
            #print(anchor_ious_all)
            #print('self ref anchors shape ',self.ref_anchors[output_id].shape)
            #print('Truth shape: ',truth_box.shape)
            best_n_all = anchor_ious_all.argmax(dim=1) # shape n_truth_box
            #print(best_n_all.shape,'\n')
            #print(best_n_all)
            #print(best_n_all)
            #print(anchor_ious_all)
            #print('Best n all:',best_n_all)
            best_n_mask = ((best_n_all == self.anch_masks[output_id][0]) | (best_n_all == self.anch_masks[output_id][1]) | (best_n_all == self.anch_masks[output_id][2]))
            #print(best_n_mask)
            # Best n mask is true or false 
            #print(self.anch_masks[output_id][0])
            #print(best_n_mask)

            if sum(best_n_mask) == 0:
                continue

            best_n = best_n_all % 3 # reduce between 0 and 3 

            #print(truth_box)

            truth_box[:n, 0] = truth_x_all[b, :n]
            truth_box[:n, 1] = truth_y_all[b, :n]
            #print(truth_box)
            #print(truth_box)
            #print(pred[b].shape)
            pred_ious = bboxes_iou(pred[b].view(-1, 4), truth_box, xyxy=False, CIoU=True)
            #print(pred[b])
            #print(pred_ious)
            #print(pred[b].view(-1, 4).shape)
            pred_best_iou, _ = pred_ious.max(dim=1)
            #print(pred[b].shape)
            #print(pred_best_iou.max())
            #print(pred_best_iou)
            pred_best_iou = (pred_best_iou > self.ignore_thre)
            #print(pred_best_iou.shape)
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3]) # shape n_anchors, fsize, fsize
            #print(pred_best_iou.shape)
            # set mask to zero (ignore) if pred matches truth
            obj_mask[b] = ~ pred_best_iou
            #print(obj_mask[b])
            #print(best_n.shape)
            #print(best_n)

            for ti in range(best_n.shape[0]):
                if best_n_mask[ti] == 1:
                    # Se il bounding box è trovato dalla risoluzione n 
                    # get the center(s)
                    i, j = truth_i[ti], truth_j[ti]
                    #print(i,j)
                    a = best_n[ti]
                    obj_mask[b, a, j, i] = 1
                    tgt_mask[b, a, j, i, :] = 1
                    # TARGET
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - truth_x_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - truth_y_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 2] = torch.log(
                        truth_w_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 0] + 1e-16)
                    target[b, a, j, i, 3] = torch.log(
                        truth_h_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 1] + 1e-16)
                    target[b, a, j, i, 4] = 1 # object in the target! 
                    target[b, a, j, i, 5 + labels[b, ti, 4].to(torch.int16).cpu().numpy()] = 1 # class! 
                    ##############################
                    tgt_scale[b, a, j, i, :] = torch.sqrt(2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)

        return obj_mask, tgt_mask, tgt_scale, target

    def forward(self, xin, labels=None):
        loss, loss_xy, loss_wh, loss_obj, loss_cls = 0, 0, 0, 0, 0
        for output_id, output in enumerate(xin):
            #print('resolution '+str(output_id)+'\n')
            batchsize = output.shape[0]
            # fsize can be the size of the output (first scaled by 8, second by 16 and so on)
            fsize = output.shape[2]
            n_ch = 5 + self.n_classes

            output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize)
            output = output.permute(0, 1, 3, 4, 2)  # .contiguous()

            # logistic activation for xy, obj, cls
            output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(output[..., np.r_[:2, 4:n_ch]])

            pred = output[..., :4].clone()
            pred[..., 0] += self.grid_x[output_id] # add the grid
            pred[..., 1] += self.grid_y[output_id] # add the grid
            pred[..., 2] = torch.exp(pred[..., 2]) * self.anchor_w[output_id] # to make non negative!
            pred[..., 3] = torch.exp(pred[..., 3]) * self.anchor_h[output_id] # to make non negative!

            obj_mask, tgt_mask, tgt_scale, target = self.build_target(pred, labels, batchsize, fsize, n_ch, output_id)
            
            # loss calculation
            output[..., 4] *= obj_mask
            output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            output[..., 2:4] *= tgt_scale

            target[..., 4] *= obj_mask
            target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            target[..., 2:4] *= tgt_scale

            loss_xy += F.binary_cross_entropy(input=output[..., :2], target=target[..., :2],
                                              weight=tgt_scale * tgt_scale, reduction='sum')
            loss_wh += F.mse_loss(input=output[..., 2:4], target=target[..., 2:4], reduction='sum') / 2
            loss_obj += F.binary_cross_entropy(input=output[..., 4], target=target[..., 4], reduction='sum')
            loss_cls += F.binary_cross_entropy(input=output[..., 5:], target=target[..., 5:], reduction='sum')

        loss = loss_xy + loss_wh + loss_obj + loss_cls

        return loss, loss_xy, loss_wh, loss_obj, loss_cls


def collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append([img])
        bboxes.append([box])
    images = np.concatenate(images, axis=0)
    images = images.transpose(0, 3, 1, 2)
    images = torch.from_numpy(images).div(255.0)
    bboxes = np.concatenate(bboxes, axis=0)
    bboxes = torch.from_numpy(bboxes)
    return images, bboxes

def train(model, device, config, epochs=5, save_cp=True, log_step=200, calc_loss_validation = True, evaluate_averages=False):
    train_dataset = Yolo_dataset(config.train_label, config, train=True)
    val_dataset = Yolo_dataset(config.val_label, config, train=False)

    n_train = len(train_dataset)

    log_step = int(n_train/config.batch/5) # Evaluate validation loss 5 times per epoch

    train_loader = DataLoader(train_dataset, batch_size=config.batch // config.subdivisions, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True, collate_fn=collate)

    val_loader = DataLoader(val_dataset, batch_size=config.batch // config.subdivisions, shuffle=True, num_workers=8,
                            pin_memory=True, drop_last=True, collate_fn=collate)

    global_step = 0
    
    # learning rate setup
    def burnin_schedule(i):
        if i < config.burn_in:
            factor = pow(i / config.burn_in, 4)
        elif i < config.steps[0]:
            factor = 1.0
        elif i < config.steps[1]:
            factor = 0.1
        else:
            factor = 0.01
        return factor

    if config.TRAIN_OPTIMIZER.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate / config.batch,
            betas=(0.9, 0.999),
            eps=1e-08,
        )
    elif config.TRAIN_OPTIMIZER.lower() == 'sgd':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config.learning_rate / config.batch,
            momentum=config.momentum,
            weight_decay=config.decay,
        )
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)

    criterion = Yolo_loss(device=device, batch=config.batch // config.subdivisions, n_classes=config.classes)

    save_prefix = 'Yolov4_epoch'
    saved_models = deque()
    model.train()
    # To evaluate model
    step_list = []
    loss_list = []

    val_loss_list = []
    val_ap = []
    val_ap_custom = []

    # Creating figure for total loss 
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.title('Loss on training samples')

    # Creating figure for validation total losses
    fig3 = plt.figure()
    ax6 = fig3.add_subplot(111)
    plt.title('Loss on validation samples')

    for epoch in range(epochs):
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img', ncols=50) as pbar:
            for i, batch in enumerate(train_loader):
                images = batch[0] # shape [batch_size, n_ch, width, height]
                bboxes = batch[1] # shape [batch_size, n_boxes, box_coord+n_classes]

                images = images.to(device=device, dtype=torch.float32)
                bboxes = bboxes.to(device=device)

                bboxes_pred = model(images) # shape [num_resolutions, batch_size, (5+n_ch)*num_boxes, grid, grid]

                loss, loss_xy, loss_wh, loss_obj, loss_cls = criterion(bboxes_pred, bboxes)
                # loss = loss / config.subdivisions
                loss.backward()

                if global_step % config.subdivisions == 0:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

                if global_step % (log_step * config.subdivisions) == 0: # After tot images, print info 
                    print('\n\nTotal loss: ',loss.item(),'\nLast learning rate: ',scheduler.get_last_lr())
                    print('Loss center bboxes: ',loss_xy.item(),'\nLoss bboxes dimension: ',loss_wh.item(),'\nLoss objectness: ',loss_obj.item(),'\nLoss class: ',loss_cls.item())
                    
                    # Calculating validation loss 
                    valid_loss = 0.0
                    if(calc_loss_validation):
                        with torch.no_grad():
                            # Creating the temporary model for evaluation
                            model.eval()
                            for i_val, batch_val in enumerate(val_loader):
                                if(i_val>10):
                                    break
                                images_val = batch_val[0] # shape [batch_size, n_ch, width, height]
                                bboxes_val = batch_val[1] # shape [batch_size, n_boxes, box_coord+n_classes]

                                images_val = images.to(device=device, dtype=torch.float32)
                                bboxes_val = bboxes.to(device=device)

                                bboxes_pred_val = model(images_val) # shape [num_resolutions, batch_size, (5+n_ch)*num_boxes, grid, grid]

                                losses = criterion(bboxes_pred_val, bboxes_val)
                                valid_loss+=losses[0].cpu().detach().numpy()
                            valid_loss/=i_val
                            val_loss_list.append(valid_loss)
                    
                        model.train()
                    


                    # Update lists 
                    loss_list.append(loss.item())
                    step_list.append(global_step)
                                    

                pbar.update(images.shape[0]) 
                global_step += 1
        
        pbar.close()

        if save_cp:
            try:
                os.makedirs(config.checkpoints, exist_ok=True)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            save_path = os.path.join(config.checkpoints, f'{save_prefix}{epoch + 1}.pth')
            torch.save(model.state_dict(), save_path)
            logging.info(f'Checkpoint {epoch + 1} saved !')
            saved_models.append(save_path)
            if len(saved_models) > config.keep_checkpoint_max > 0:
                model_to_remove = saved_models.popleft()
                try:
                    os.remove(model_to_remove)
                except:
                    logging.info(f'failed to remove {model_to_remove}')

        # Calculate ap ar
        if(evaluate_averages):
            print('\nEpoch: ', epoch)
            print('Evaluating averages')
            # Create a model 
            with torch.no_grad():
                model_eval = Yolov4(yolov4conv137weight=None,n_classes=config.classes,inference=True)
                device = torch.device('cuda')
                model_eval.load_state_dict(model.state_dict())
                model_eval.to(device=device)
                detector = Detector(model_eval,True,config.width,config.height,config.val_dataset_dir,keep_aspect_ratio=False)
                metric_obj = Metric(config.val_label,config.val_dataset_dir)
                pred,_ = detector.detect_in_images(0.01)
                confidence_steps = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
                values = metric_obj.calculate_precision_recall_f1_lists(pred,confidence_steps,0.3)
                # AP calc.
                average_prec = metric_obj.calc_AP(values[0],values[1])
                val_ap.append(average_prec)
                # # on custom dataset 
                # detector = Detector(model_eval,True,config.width,config.height,config.custom_val_dir,keep_aspect_ratio=False)
                # metric_obj = Metric(config.custom_val_label,config.custom_val_dir)
                # pred,_ = detector.detect_in_images(0.01)
                # values = metric_obj.calculate_precision_recall_f1_lists(pred,confidence_steps,0.3)
                # # AP calc.
                # average_prec = metric_obj.calc_AP(values[0],values[1])
                # val_ap_custom.append(average_prec)
        
        
    if(evaluate_averages):
        # PLOT GRAPH AP AND AR
        fig4 = plt.figure()
        ax7 = fig4.add_subplot(111)
        ax7.plot(range(epochs),val_ap)
        plt.title('Average precision on training dataset validation')

        # fig5 = plt.figure()
        # ax8 = fig5.add_subplot(111)
        # ax8.plot(range(epochs),val_ap_custom)
        # plt.title('Average precision on custom dataset validation')

    # PLOT GRAPHS LOSSES
    ax1.plot(step_list,loss_list)
    ax6.plot(step_list,val_loss_list)

    plt.show()



def get_args(**kwargs):
    cfg = kwargs
    return edict(cfg)


if __name__ == '__main__':
    cfg = get_args(**Cfg)

    # Fine tuning starting from yolo pretrained weights
    weight_path = r'C:\Users\Melgani\Desktop\master_degree\weight\yolov4.pth'
    device = torch.device('cuda')
    # Creating the empty model
    model = Yolov4(yolov4conv137weight=None,n_classes=1)
    # Fusing dictionaries of weights, all the weights except the heads 
    new_dictionary = {}
    weight_dictionary = torch.load(weight_path)
    for key, value in weight_dictionary.items():
        if(not ('head.conv2' in key or 'head.conv10' in key or 'head.conv18' in key)):
            new_dictionary[key]=value
    
    #Add the remaining keys
    model_dict = model.state_dict()
    for key, value in model_dict.items():
        if (not key in new_dictionary.keys()):
            new_dictionary[key] = value

    # Loading the dict 
    model.load_state_dict(new_dictionary)
    model.to(device=device)
    
    # FIRST PHASE
    # Freeze all the layers except the changed ones 
    print('Freezing layers..')
    for name, param in model.named_parameters():
        if(not ('head.conv2' in name or 'head.conv10' in name or 'head.conv18' in name)):
           param.requires_grad = False
    
    Cfg.learning_rate = 0.001
    cfg.TRAIN_EPOCHS = 20
    
    train(model=model,config=cfg,epochs=cfg.TRAIN_EPOCHS,device=device,calc_loss_validation=True, save_cp=True,evaluate_averages=True)

    # SECOND PHASE
    print('Unfreezing backbone and neck layers..')
    for name, param in model.named_parameters():
        param.requires_grad = True
    
    Cfg.learning_rate = 0.0005
    cfg.TRAIN_EPOCHS = 40

    train(model=model,config=cfg,epochs=cfg.TRAIN_EPOCHS,device=device,calc_loss_validation=True, save_cp=True,evaluate_averages=True)

    # THIRD PHASE
    print('Freezing backbone and neck layers..')
    for name, param in model.named_parameters():
        if(not 'head' in name):
            param.requires_grad = False
    
    Cfg.learning_rate = 0.0005
    cfg.TRAIN_EPOCHS = 5

    train(model=model,config=cfg,epochs=cfg.TRAIN_EPOCHS,device=device,calc_loss_validation=True, save_cp=True, evaluate_averages=True)

    # Saving the weights 
    save_path = os.path.join(cfg.savings_path, f'{cfg.dataset_name}{cfg.width}.pth')
    torch.save(model.state_dict(), save_path)
