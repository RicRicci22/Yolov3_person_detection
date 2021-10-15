from hashlib import new
import os
import random
from copy import deepcopy

import cv2
import numpy as np

from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt

def rand_uniform_strong(min, max):
    return random.randint(min*100,max*100)/100 


def fill_truth_detection(bboxes, num_boxes, classes, left_extreme,right_extreme, top_extreme,bottom_extreme, original_width,original_height,net_w, net_h):
    if bboxes.shape[0] == 0:
        return bboxes, 10000
    np.random.shuffle(bboxes)

    bboxes[:, 0] = np.clip(bboxes[:, 0], left_extreme, right_extreme)
    bboxes[:, 2] = np.clip(bboxes[:, 2], left_extreme, right_extreme)

    bboxes[:, 1] = np.clip(bboxes[:, 1], top_extreme, bottom_extreme)
    bboxes[:, 3] = np.clip(bboxes[:, 3], top_extreme, bottom_extreme)

    out_box = list(np.where(((bboxes[:, 1] == bottom_extreme) & (bboxes[:, 3] == bottom_extreme)) |
                            ((bboxes[:, 0] == right_extreme) & (bboxes[:, 2] == right_extreme)) |
                            ((bboxes[:, 1] == top_extreme) & (bboxes[:, 3] == top_extreme)) |
                            ((bboxes[:, 0] == left_extreme) & (bboxes[:, 2] == left_extreme)))[0])
    list_box = list(range(bboxes.shape[0]))
    for i in out_box:
        list_box.remove(i)
    bboxes = bboxes[list_box]

    if bboxes.shape[0] == 0:
        return bboxes
    

    bboxes = bboxes[np.where((bboxes[:, 4] < classes) & (bboxes[:, 4] >= 0))[0]]

    if bboxes.shape[0] > num_boxes:
        bboxes = bboxes[:num_boxes]

    # Sx is original width, sy is original height
    bboxes[:, 0] *= (net_w / original_width)
    bboxes[:, 2] *= (net_w / original_width)
    bboxes[:, 1] *= (net_h / original_height)
    bboxes[:, 3] *= (net_h / original_height)

    return bboxes


def image_data_augmentation(mat, w, h, flip, dhue, dsat, dexp, gaussian_noise, blur,truth):
    #try:
    img = mat
    #oh, ow, _ = img.shape

    sized = cv2.resize(img, (w, h), cv2.INTER_LINEAR)

    # flip
    if flip==1:
        sized = cv2.flip(sized, flip) 
    elif flip==-1:
        sized = cv2.flip(sized,flip)
    elif flip==0:
        sized = cv2.flip(sized,flip)

    # HSV augmentation
    # cv2.COLOR_BGR2HSV, cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2BGR, cv2.COLOR_HSV2RGB
    if dsat != 1 or dexp != 1 or dhue != 0:
        if img.shape[2] >= 3:
            hsv_src = cv2.cvtColor(sized.astype(np.float32), cv2.COLOR_RGB2HSV)  # RGB to HSV
            hsv = cv2.split(hsv_src)
            hsv[1] *= dsat
            hsv[2] *= dexp
            hsv[0] += 179 * dhue
            hsv_src = cv2.merge(hsv)
            sized = np.clip(cv2.cvtColor(hsv_src, cv2.COLOR_HSV2RGB), 0, 255)  # HSV to RGB (the same as previous)
        else:
            sized *= dexp
    if blur:
        if blur == 1:
            dst = cv2.GaussianBlur(sized, (7, 7), 0)
            for b in truth:
                dst[int(b[0]):int(b[2]),int(b[1]):int(b[3]),:] = sized[int(b[0]):int(b[2]),int(b[1]):int(b[3]),:]
        else:
            dst = cv2.GaussianBlur(sized, (7, 7), 0)

        sized = dst

    # Gaussian noise 
    if gaussian_noise:
        # Could not be negative the variance
        sigma = gaussian_noise**0.5
        gauss = np.random.normal(0,sigma,sized.shape)
        gauss = gauss*255
        # Adding the two images
        sized = sized + gauss
        # Clipping between 0 and 255
        sized = np.clip(sized,0,255)
    # except:
    #     print("OpenCV can't augment image!")
    #     sized = mat

    return sized.astype('float32')


def filter_truth(bboxes, dx, dy, sx, sy, xd, yd):
    bboxes[:, 0] -= dx
    bboxes[:, 2] -= dx
    bboxes[:, 1] -= dy
    bboxes[:, 3] -= dy

    bboxes[:, 0] = np.clip(bboxes[:, 0], 0, sx)
    bboxes[:, 2] = np.clip(bboxes[:, 2], 0, sx)

    bboxes[:, 1] = np.clip(bboxes[:, 1], 0, sy)
    bboxes[:, 3] = np.clip(bboxes[:, 3], 0, sy)

    out_box = list(np.where(((bboxes[:, 1] == sy) & (bboxes[:, 3] == sy)) |
                            ((bboxes[:, 0] == sx) & (bboxes[:, 2] == sx)) |
                            ((bboxes[:, 1] == 0) & (bboxes[:, 3] == 0)) |
                            ((bboxes[:, 0] == 0) & (bboxes[:, 2] == 0)))[0])
    list_box = list(range(bboxes.shape[0]))
    for i in out_box:
        list_box.remove(i)
    bboxes = bboxes[list_box]

    bboxes[:, 0] += xd
    bboxes[:, 2] += xd
    bboxes[:, 1] += yd
    bboxes[:, 3] += yd

    return bboxes


def blend_truth_mosaic(out_img, img, bboxes, w, h, cut_x, cut_y, i_mixup):
    
    left_shift =  w - cut_x
    top_shift = h - cut_y
    right_shift = cut_x
    bot_shift = cut_y

    if i_mixup == 0:
        bboxes = filter_truth(bboxes, left_shift, top_shift, cut_x, cut_y, 0, 0)
        out_img[:cut_y, :cut_x] = img[top_shift:top_shift + cut_y, left_shift:left_shift + cut_x]
    if i_mixup == 1:
        bboxes = filter_truth(bboxes, cut_x - right_shift, top_shift, w - cut_x, cut_y, cut_x, 0)
        out_img[:cut_y, cut_x:] = img[top_shift:top_shift + cut_y, cut_x - right_shift:w - right_shift]
    if i_mixup == 2:
        bboxes = filter_truth(bboxes, left_shift, cut_y - bot_shift, cut_x, h - cut_y, 0, cut_y)
        out_img[cut_y:, :cut_x] = img[cut_y - bot_shift:h - bot_shift, left_shift:left_shift + cut_x]
    if i_mixup == 3:
        bboxes = filter_truth(bboxes, cut_x - right_shift, cut_y - bot_shift, w - cut_x, h - cut_y, cut_x, cut_y)
        out_img[cut_y:, cut_x:] = img[cut_y - bot_shift:h - bot_shift, cut_x - right_shift:w - right_shift]

    return out_img, bboxes


def draw_box(img, bboxes):
    for b in bboxes:
        img = cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
    return img


class Yolo_dataset(Dataset):
    def __init__(self, lable_path, cfg, train=True):
        super(Yolo_dataset, self).__init__()
        if cfg.mixup == 2:
            print("cutmix=1 - isn't supported for Detector")
            raise
        elif cfg.mixup == 2 and cfg.letter_box:
            print("Combination: letter_box=1 & mosaic=1 - isn't supported, use only 1 of these parameters")
            raise

        self.cfg = cfg
        self.train = train

        truth = {}
        f = open(lable_path, 'r', encoding='utf-8')
        for line in f.readlines():
            data = line.split(" ")
            truth[data[0]] = []
            for i in data[1:]:
                truth[data[0]].append([int(float(j)) for j in i.split(',')])

        self.truth = truth
        self.imgs = list(self.truth.keys())

    def __len__(self):
        return len(self.truth.keys())

    def __getitem__(self, index):
        if not self.train:
            return self._get_val_item(index)
        img_path = self.imgs[index]
        bboxes = np.array(self.truth.get(img_path), dtype=np.float)
        img_path = os.path.join(self.cfg.train_dataset_dir, img_path)
        aug_variable = 0
        # Creating augmentation variable
        # Both methods selected
        if(self.cfg.mosaic and self.cfg.mixup):
            # Choose one between mosaic and mixup 
            temp = random.randint(0,1)
            if temp:
                # Mosaic 
                aug_variable = 3
            else:
                # Mixup 
                aug_variable = 1 
        else:
            if(self.cfg.mosaic):
                # Mosaic 
                aug_variable = 3
            
            if(self.cfg.mixup):
                # Mixup  
                aug_variable = 1
        
        # Randomly disable mosaic and mixup 
        if random.randint(0, 1):
            aug_variable = 0

        if aug_variable == 3:
            min_offset = 0.2
            cut_x = random.randint(int(self.cfg.width * min_offset), int(self.cfg.width * (1 - min_offset)))
            cut_y = random.randint(int(self.cfg.height * min_offset), int(self.cfg.height * (1 - min_offset)))

        dhue, dsat, dexp, flip, blur, rot, crop = 0, 1, 1, 0, 0, -1, 0
        gaussian_noise = 0

        out_img = np.zeros([self.cfg.height, self.cfg.width, 3])
        out_bboxes = []

        for i in range(aug_variable + 1):
            if i != 0:
                img_path = random.choice(list(self.truth.keys()))
                bboxes = np.array(self.truth.get(img_path), dtype=np.float)
                img_path = os.path.join(self.cfg.train_dataset_dir, img_path)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img is None:
                continue
            oh, ow, _ = img.shape

            if(self.cfg.hue and random.randint(0,1)):
                dhue = rand_uniform_strong(-0.5, 0.5)
            if(self.cfg.saturation and random.randint(0,1)):
                dsat = rand_uniform_strong(0, 2)
            if(self.cfg.exposure and random.randint(0,1)):
                dexp = rand_uniform_strong(0.5, 2)

            if(self.cfg.flip and random.randint(0,1)):
                flip = random.randint(-1,1)
            else:
                flip = -2
            
            if((self.cfg.rotate) and random.randint(0,1)):
                rot = random.randint(0,1)

            if(self.cfg.crop and random.randint(0,1)):
                crop = 1
                # Crop image 
                random_x = 0
                random_y = 0 

                if(ow>self.cfg.width):
                    # Can crop on width
                    possible_x_positions = [int(self.cfg.width/2)+1+i for i in range(0,ow-self.cfg.width-2)]
                    random_x_index = random.randint(0,len(possible_x_positions)-1)
                    random_x = possible_x_positions[random_x_index]
                if(oh>self.cfg.height):
                    possible_y_positions = [int(self.cfg.height/2)+1+i for i in range(0,oh-self.cfg.height-2)]
                    random_y_index = random.randint(0,len(possible_y_positions)-1)
                    random_y = possible_y_positions[random_y_index]
                
                if(random_x!=0 and random_y!=0):
                    # Crop image
                    new_img = img[random_y-int(self.cfg.height/2):random_y+int(self.cfg.height/2),random_x-int(self.cfg.width/2):random_x+int(self.cfg.width/2)]
                    img = new_img
                    oh, ow, _ = img.shape
                    # Check boxes 
                    left_extreme = random_x-int(self.cfg.width/2)
                    right_extreme = random_x+int(self.cfg.width/2)
                    top_extreme = random_y-int(self.cfg.height/2)
                    bottom_extreme = random_y+int(self.cfg.height/2)
                    truth = fill_truth_detection(bboxes, self.cfg.boxes, self.cfg.classes, left_extreme, right_extreme, top_extreme,bottom_extreme, ow,oh, self.cfg.width, self.cfg.height)
                    truth[:,0]-=left_extreme
                    truth[:,2]-=left_extreme
                    truth[:,1]-=top_extreme
                    truth[:,3]-=top_extreme
                elif(random_x==0 and random_y!=0):
                    new_img = img[random_y-int(self.cfg.height/2):random_y+int(self.cfg.height/2),:]
                    img = new_img
                    oh, ow, _ = img.shape
                    # Check boxes 
                    top_extreme = random_y-int(self.cfg.height/2)
                    bottom_extreme = random_y+int(self.cfg.height/2)
                    truth = fill_truth_detection(bboxes, self.cfg.boxes, self.cfg.classes, 0,ow, top_extreme,bottom_extreme, ow,oh, self.cfg.width, self.cfg.height)
                    truth[:,1]-=top_extreme
                    truth[:,3]-=top_extreme
                elif(random_x!=0 and random_y==0):
                    new_img = img[:,random_x-int(self.cfg.width/2):random_x+int(self.cfg.width/2)]
                    img = new_img
                    oh, ow, _ = img.shape
                    # Check boxes 
                    left_extreme = random_x-int(self.cfg.width/2)
                    right_extreme = random_x+int(self.cfg.width/2)
                    truth = fill_truth_detection(bboxes, self.cfg.boxes, self.cfg.classes, left_extreme,right_extreme, 0,oh, ow,oh, self.cfg.width, self.cfg.height)
                    truth[:,0]-=left_extreme
                    truth[:,2]-=left_extreme
                else:
                    truth = fill_truth_detection(bboxes, self.cfg.boxes, self.cfg.classes, 0, ow, 0,oh, ow,oh, self.cfg.width, self.cfg.height)
        
            
            swidth = ow 
            sheight = oh

            if(not crop):
                truth = fill_truth_detection(bboxes, self.cfg.boxes, self.cfg.classes,0, swidth, 0,sheight, ow,oh, self.cfg.width, self.cfg.height)

            # FLIPPING 
            if flip==-1:
                temp_width = self.cfg.width - truth[:, 0]
                temp_height = self.cfg.height - truth[:, 1]
                truth[:, 0] = self.cfg.width - truth[:, 2]
                truth[:, 1] = self.cfg.height - truth[:, 3]
                truth[:, 2] = temp_width
                truth[:, 3] = temp_height
            elif flip==1:
                temp_width = self.cfg.width - truth[:, 0]
                truth[:, 0] = self.cfg.width - truth[:, 2]
                truth[:, 2] = temp_width
            elif(flip==0):
                temp_height = self.cfg.height - truth[:, 1]
                truth[:, 1] = self.cfg.height - truth[:, 3]
                truth[:, 3] = temp_height


            if (self.cfg.blur):
                blur = random.randint(0,2)  # 0 - disable, 1 - blur background, 2 - blur the whole image

            # Setting gaussian noise 
            if self.cfg.gaussian_noise and random.randint(0, 1):
                gaussian_noise = random.randint(1,10)/1000
            
            ai = image_data_augmentation(img, self.cfg.width, self.cfg.height, flip, dhue, dsat, dexp, gaussian_noise, blur, truth)

            oh,ow,_ = ai.shape

            if(rot==1):
                # rotate 90 degrees clockwise
                new_truth = deepcopy(truth)
                ai = cv2.rotate(ai, cv2.ROTATE_90_CLOCKWISE)
                # Boxes
                new_truth[:,0] = oh-truth[:,3]
                new_truth[:,1] = truth[:,0]
                new_truth[:,2] = oh-truth[:,1]
                new_truth[:,3] = truth[:,2]
                truth = new_truth
            elif(rot==0):
                # rotate 90 degrees counterclockwise
                new_truth = deepcopy(truth)
                ai = cv2.rotate(ai, cv2.ROTATE_90_COUNTERCLOCKWISE)
                # Boxes
                new_truth[:,0] = truth[:,1]
                new_truth[:,1] = ow-truth[:,2]
                new_truth[:,2] = truth[:,3]
                new_truth[:,3] = ow-truth[:,0]
                truth = new_truth

            if aug_variable == 0:
                # NOTHING
                out_img = ai
                out_bboxes = truth
            if aug_variable == 1:
                # MIXUP
                if i == 0:
                    old_img = ai.copy()
                    old_truth = truth.copy()
                elif i == 1:
                    out_img = cv2.addWeighted(ai, 0.5, old_img, 0.5, 0.2)
                    out_bboxes = np.concatenate([old_truth, truth], axis=0)
            elif aug_variable == 3:
                # MOSAIC
                out_img, out_bbox = blend_truth_mosaic(out_img, ai, truth.copy(), self.cfg.width, self.cfg.height, cut_x,
                                                       cut_y, i)
                out_bboxes.append(out_bbox)
                
        if aug_variable == 3:
            out_bboxes = np.concatenate(out_bboxes, axis=0)

        out_bboxes1 = np.zeros([self.cfg.boxes, 5])
        out_bboxes1[:min(out_bboxes.shape[0], self.cfg.boxes)] = out_bboxes[:min(out_bboxes.shape[0], self.cfg.boxes)]
        return out_img, out_bboxes1

    def _get_val_item(self, index):
        # Get validation item
        img_path = self.imgs[index]
        boxes = np.zeros((self.cfg.boxes,5))
        boxes_to_insert = np.array(self.truth.get(img_path), dtype=np.float)
        img = cv2.imread(os.path.join(self.cfg.val_dataset_dir, img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_height, original_width, _ = img.shape
        
        # Resizing
        img = cv2.resize(img, (self.cfg.width, self.cfg.height), cv2.INTER_LINEAR) 

        # Resize bbox to new width height
        boxes_to_insert[:, 0] *= (self.cfg.width / original_width)
        boxes_to_insert[:, 2] *= (self.cfg.width / original_width)
        boxes_to_insert[:, 1] *= (self.cfg.height / original_height)
        boxes_to_insert[:, 3] *= (self.cfg.height / original_height)


        if(boxes_to_insert.shape[0]>self.cfg.boxes):
            boxes = boxes_to_insert[:self.cfg.boxes,:]
        else:
            boxes[:boxes_to_insert.shape[0]] = boxes_to_insert

        return img, boxes


if __name__ == "__main__":
    from cfg import Cfg
    import matplotlib.pyplot as plt

    random.seed(2020)
    np.random.seed(2020)
    dataset = Yolo_dataset(Cfg.train_label, Cfg, train=True)
    for i in range(100):
        out_img, out_bboxes = dataset.__getitem__(i)
        a = draw_box(out_img.copy(), out_bboxes.astype(np.int32))
        plt.imshow(out_img.astype(np.int32))
        plt.show()
