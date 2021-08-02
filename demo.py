# -*- coding: utf-8 -*-
'''
@Time          : 20/04/25 15:49
@Author        : huguanghao
@File          : demo.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :
'''

# import sys
# import time
# from PIL import Image, ImageDraw
# from models.tiny_yolo import TinyYoloNet
from tool.utils import *
from tool.darknet2pytorch import Darknet
import argparse

"""hyper parameters"""
use_cuda = True
num_classes = 80
if num_classes == 20:
    namesfile = 'data/voc.names'
elif num_classes == 80:
    namesfile = 'data/coco.names'
else:
    namesfile = 'data/x.names'


def process_all_videos(cfgfile, weightfile, video_folder):
    # Script that automatically process the same 2 videos using different confidence thresholds
    import cv2
    import os
    import fnmatch

    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()
        print('GPU activated!')

    # Enter the folder with the two videos and get only .mp4 files
    videos = fnmatch.filter(os.listdir(video_folder), '*.mp4')
    i=0
    for video in videos:
        i=i+1
        print('Processing video',i,'out of',len(videos))
        cap = cv2.VideoCapture(video_folder + '/' + video)
        # Get info on video
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cap.set(3, 1280)
        cap.set(4, 720)
        print("Starting the YOLO loop...")
        print(m.width)
        print(m.height)
        # Loading class names
        class_names = load_class_names(namesfile)
        # Object to save the video
        width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer= cv2.VideoWriter(video_folder+'/result '+video, 0x00000021, fps, (width,height))
        n_frame = 1
        start = time.time()
        while (n_frame<=frame_count):
            print('Processing frame',n_frame,'out of',frame_count)
            ret, img = cap.read()
            sized = cv2.resize(img, (m.width, m.height))
            sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

            #start = time.time()
            boxes = do_detect(m, sized, 0.5, num_classes, 0.4, use_cuda)
            #finish = time.time()
            #print('Predicted in %f seconds.' % (finish - start))

            result_img = plot_boxes_cv2(img, boxes, savename=None, class_names=class_names)

            # cv2.imshow('Yolo demo', result_img)
            writer.write(result_img)
            cv2.waitKey(1)
            n_frame=n_frame+1
        finish = time.time()
        # Printing average FPS in a txt file inside the folder
        f = open(video_folder+"/average_FPS.txt", "a")
        f.write(video+'-->'+str(round(frame_count/(finish-start),1))+'FPS\n')
        f.close
        cap.release()
        writer.release()

def detect_in_images(dataset_input,cfgfile,weightfile,confidence):
    import cv2
    # This function will create a file in the same folder as the dataset containing the predictions
    # dataset_input = folder containing all the images where the detection has to be performed
    # HYPERPARAMETERS
    use_cuda = True
    height = 1024
    width = 1024
    detections = open(dataset_input+r'\_predictions.txt','w')
    m = Darknet(cfgfile)
    m.load_weights(weightfile)
    if use_cuda:
        m.cuda()
        print('GPU activated!')
    for filename in os.listdir(dataset_input):
        print('Processing image '+filename+'\n')
        f = os.path.join(dataset_input, filename)
        if(os.path.isfile(f)):
            # The file is an image
            detections.write(filename)
            img = cv2.imread(os.path.join(dataset_input,filename))
            original_height, original_width, _ = img.shape
            sized = cv2.resize(img, (width, height))
            sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
            # perform detection
            boxes = do_detect(m, sized, confidence, num_classes, 0.4, use_cuda)
            for box in boxes:
                if(box[6]==0):
                    x1 = int((box[0] - box[2] / 2.0) * original_width)
                    y1 = int((box[1] - box[3] / 2.0) * original_height)
                    x2 = int((box[0] + box[2] / 2.0) * original_width)
                    y2 = int((box[1] + box[3] / 2.0) * original_height)
                    # check if negative
                    if(x1<0):
                        x1=0
                    if(y1<0):
                        y1=0
                    if(x2<0):
                        x2=0
                    if(y2<0):
                        y2=0
                    obj_class = box[6] #0 means person
                    print(x1,y1,x2,y2,obj_class)
                    # Insert prediction
                    detections.write(' '+str(x1)+','+str(y1)+','+str(x2)+','+str(y2)+','+str(obj_class))
            detections.write('\n')


def process_videos_different_confidence(cfgfile, weightfile, video_path, confidences):
    # Script that automatically process the same 2 videos using different confidence thresholds
    # confidences -> LIST of confidence values
    import cv2
    import fnmatch

    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()
        print('GPU activated!')

    # Break path into pieces
    pieces = os.path.split(video_path)

    f = open(pieces[0]+"/average_FPS.txt", "a")
    f.write('\nFPS FOR DIFFERENT CONFIDENCE VALUES\n')

    for threshold in confidences:
        cap = cv2.VideoCapture(video_path)
        # Get info on video
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cap.set(3, 1280)
        cap.set(4, 720)
        print("Starting the YOLO loop...")
        print(m.width)
        print(m.height)
        # Loading class names
        class_names = load_class_names(namesfile)
        # Object to save the video
        width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer= cv2.VideoWriter(pieces[0]+'/result_'+str(threshold)+'_'+pieces[1], 0x00000021, fps, (width,height))
        n_frame = 1
        start = time.time()
        while (n_frame<=frame_count):
            print('Processing frame',n_frame,'out of',frame_count)
            ret, img = cap.read()
            sized = cv2.resize(img, (m.width, m.height))
            sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

            #start = time.time()
            boxes = do_detect(m, sized, threshold, num_classes, 0.4, use_cuda)
            #finish = time.time()
            #print('Predicted in %f seconds.' % (finish - start))

            result_img = plot_boxes_cv2(img, boxes, savename=None, class_names=class_names)

            # cv2.imshow('Yolo demo', result_img)
            writer.write(result_img)
            cv2.waitKey(1)
            n_frame=n_frame+1
        finish = time.time()
        # Printing average FPS in a txt file inside the folder
        f.write(pieces[1]+'_confidence_'+str(threshold)+'-->'+str(round(frame_count/(finish-start),1))+'FPS\n')
        f.close
        cap.release()
        writer.release()
        # Compress video
        print('Compressing video to 360p..')
        compress_video(pieces[0]+'//result_'+str(threshold)+'_'+pieces[1])

def compress_video(to_compress_path):
    import moviepy.editor as mp
    from moviepy.editor import vfx

    clip = mp.VideoFileClip(to_compress_path)
    clip_resized = clip.fx( vfx.resize, width = 1440)
    clip_resized.write_videofile(to_compress_path+'_compressed')
    clip.close()
    os.remove(to_compress_path)
    os.rename(to_compress_path+'_compressed',to_compress_path)



def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str, default='./cfg/yolov4.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        default='./checkpoints/Yolov4_epoch1.pth',
                        help='path of trained model.', dest='weightfile')
    parser.add_argument('-imgfile', type=str,
                        default='./data/mscoco2017/train2017/190109_180343_00154162.jpg',
                        help='path of your image file.', dest='imgfile')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    #if args.imgfile:
        #detect(args.cfgfile, args.weightfile, args.imgfile)
        # detect_imges(args.cfgfile, args.weightfile)
        # detect_cv2(args.cfgfile, args.weightfile, args.imgfile)
        # detect_skimage(args.cfgfile, args.weightfile, args.imgfile)
    #else:
    #detect_cv2_camera(args.cfgfile, args.weightfile)
    #detect_cv2_camera('cfg/yolov4.cfg','weight/yolov4.weights')
    #process_videos_different_confidence('cfg/yolov4.cfg','weight/yolov4.weights',r'C:\Users\farid.melgani\Desktop\riccimasterthesis\video\Zona 16.mp4',[0.2,0.3,0.4,0.5])
    #compress_video('C:/Users/farid.melgani/Desktop/riccimasterthesis/video/result_0.4_Zona 16.mp4','C:/Users/farid.melgani/Desktop/riccimasterthesis/video/result_0.4_Zona 16_compressed.mp4')
    #detect_in_images(r'C:\Users\farid.melgani\Desktop\riccimasterthesis\visdrone\test','cfg/yolov4.cfg','weight/yolov4.weights',0.5)
    #detect('cfg/yolov4.cfg','weight/yolov4.weights',r'C:\Users\farid.melgani\Desktop\riccimasterthesis\pytorch-YOLOv4\data\0000006_00159_d_0000001.jpg')
    print('hey')