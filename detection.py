# This file will contain functions to perform detection on various files such as images, videos, etc.

from metrics import Metric
from tool.utils import *
from tool.torch_utils import *
from models import Yolov4
import cv2
import os
import time

import pickle

class Detector:
    def __init__(self,model,use_cuda,input_width,input_height,testset):
        self.model = model
        self.use_cuda = use_cuda
        self.testset = testset # Path to the folder containing the testset over which to perform detections
        self.input_width = input_width
        self.input_height = input_height
        # cfgfile can be None cause it's used only when using Darknet configuration file!
        # Find namesfile
        self.namesfile = 'data/custom_names.names'

    def __str__(self):
        print('Detector object')
        print('Num. classes: 1')
        print('Testset path: ',self.testset)
        print('Model: ',self.model)
        if(self.cfgfile):
            print('Cfgfile path: ',self.cfgfile)

        return 'Cuda acceleration?: '+str(self.use_cuda)


    # def process_all_videos(self, video_folder, confidence):
    #     import os
    #     import fnmatch

    #     # Enter the folder with the two videos and get only .mp4 files
    #     videos = fnmatch.filter(os.listdir(video_folder), '*.mp4')
    #     i=0
    #     for video in videos:
    #         i=i+1
    #         print('Processing video',i,'out of',len(videos))
    #         cap = cv2.VideoCapture(video_folder + '/' + video)
    #         # Get info on video
    #         fps = cap.get(cv2.CAP_PROP_FPS)
    #         frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #         cap.set(3, 1280)
    #         cap.set(4, 720)
    #         print("Starting the YOLO loop...")
    #         print(m.width)
    #         print(m.height)
    #         # Loading class names
    #         class_names = load_class_names(self.namesfile)
    #         # Object to save the video
    #         width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #         height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #         writer= cv2.VideoWriter(video_folder+'/result '+video, 0x00000021, fps, (width,height))
    #         n_frame = 1
    #         start = time.time()
    #         while (n_frame<=frame_count):
    #             print('Processing frame',n_frame,'out of',frame_count)
    #             ret, img = cap.read()
    #             sized = cv2.resize(img, (m.width, m.height))
    #             sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    #             #start = time.time()
    #             boxes = do_detect(self.model, sized, confidence, self.num_classes, 0.4, self.use_cuda)
    #             #finish = time.time()
    #             #print('Predicted in %f seconds.' % (finish - start))

    #             result_img = plot_boxes_cv2(img, boxes, savename=None, class_names=class_names)

    #             # cv2.imshow('Yolo demo', result_img)
    #             writer.write(result_img)
    #             cv2.waitKey(1)
    #             n_frame=n_frame+1
    #         finish = time.time()
    #         # Printing average FPS in a txt file inside the folder
    #         # TODO
    #         f = open(video_folder+"/average_FPS.txt", "a")
    #         f.write(video+'-->'+str(round(frame_count/(finish-start),1))+'FPS\n')
    #         f.close
    #         cap.release()
    #         writer.release()

    def detect_in_images(self, confidence, keep_aspect_ratio=False, output_file=False):
        # Perform prediction using yolov4 pytorch implementation
        # Creating the file to save output
        if(output_file):
            detections = open(self.testset+r'\_predictions.txt','w')
        # Create output prediction dictionary
        predictions_dict = {}
        # Starting the loop
        i=0
        t0 = time.time()
        #print('Detecting in images..')
        for filename in os.listdir(self.testset):
            f = os.path.join(self.testset, filename)
            if(os.path.isfile(f)):
                # Is the file an image??
                if(filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))):
                    #print('Processing image '+filename)
                    i +=1
                    img = cv2.imread(os.path.join(self.testset,filename))
                    original_height, original_width, _ = img.shape
                    if(keep_aspect_ratio):
                        sized = keep_ratio(img,self.input_height)
                        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
                    else:
                        sized = cv2.resize(img, (self.input_width, self.input_height))
                        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
                    # perform detection
                    boxes = do_detect(self.model, sized, confidence, 0.4, self.use_cuda,print_time=False)
                    # Process boxes to keep only people (boxes[0]) cause in detection batch = 1!!!!
                    new_boxes = [box for box in boxes[0] if box[5]==0]
                    if(output_file):
                        detections.write(filename)
                    predictions_dict[filename] = []
                    for box in new_boxes:
                        x1 = int(box[0]*original_width)
                        y1 = int(box[1]*original_height)
                        x2 = int(box[2]*original_width)
                        y2 = int(box[3]*original_height)
                        # check if negative
                        if(x1<0):
                            x1=0
                        if(y1<0):
                            y1=0
                        if(x2>original_width):
                            x2=original_width
                        if(y2>original_height):
                            y2=original_height
                        obj_conf = box[4]
                        obj_class = box[5] # 0 means person
                        
                        # Insert prediction
                        if(output_file):
                            detections.write(' '+str(x1)+','+str(y1)+','+str(x2)+','+str(y2)+','+str(obj_class)+','+str(obj_conf))
                        predictions_dict[filename].append([x1,y1,x2,y2,obj_class,obj_conf])
                    if(output_file):
                        detections.write('\n')
        t1 = time.time()
        print('FPS = ' + str(i/(t1-t0)))
        return predictions_dict

    def visualize_predictions(self, predictions):
        # This function will visualize all the bounding boxes for the prediction made in the testset. It will save the images with predictions in a new folder "predictions"
        print('Visualizing predictions..')
        if(not os.path.isdir('predictions')):
            os.mkdir('predictions')
        for image in os.listdir(self.testset):
            if(image.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))):
                image_cv = cv2.imread(os.path.join(self.testset,image))
            else:
                continue
            color = (0,255,0)
            imgHeight, imgWidth, _ = image_cv.shape
            thick = int((imgHeight + imgWidth) // 900)

            boxes_predicted = predictions[image]
            for box in boxes_predicted:
                left = box[0]
                top = box[1]
                right = box[2]
                bottom = box[3]
                cv2.rectangle(image_cv,(left, top), (right, bottom), color, thick)
            cv2.imwrite('predictions/drawn_'+image, image_cv)


if __name__ == '__main__':
    # PYTORCH
    # Creating the model
    model = Yolov4(yolov4conv137weight=None,n_classes=1,inference=True)
    model.load_weights(r'C:\Users\Melgani\Desktop\master_degree\weight\testino.pth')
    model.activate_gpu()

    # # Creating the detector
    yolov4_detector = Detector(model,True,608,608,r'datasets\visdrone\test2')
    pred = yolov4_detector.detect_in_images(0.8)
    yolov4_detector.visualize_predictions(pred)

    meter = Metric(r'datasets\visdrone\test2\_annotations.txt',r'datasets\visdrone\test2')
    metriche = meter.precision_recall(pred,0.5)
    #precision_list, recall_list, small_prec, small_rec, medium_prec, medium_rec, large_prec, large_rec = meter.calculate_precision_recall_curve(pred,0.5, plot_graph=True)
    print('TOTAL')
    print('Precision: '+str(metriche[0]))
    print('Recall: '+str(metriche[1]))
    print('F1: '+str(metriche[2]))
    print('\nSmall objects')
    print('Precision: '+str(metriche[3]))
    print('Recall: '+str(metriche[4]))
    print('F1: '+str(metriche[5]))
    print('\nMedium objects')
    print('Precision: '+str(metriche[6]))
    print('Recall: '+str(metriche[7]))
    print('F1: '+str(metriche[8]))
    print('\nLarge objects')
    print('Precision: '+str(metriche[9]))
    print('Recall: '+str(metriche[10]))
    print('F1: '+str(metriche[11]))
    
    # ap, ar = meter.calc_AP_AR(precision_list,recall_list)
    # aps, ars = meter.calc_AP_AR(small_prec,small_rec)
    # apm, arm = meter.calc_AP_AR(medium_prec,medium_rec)
    # apl, arl = meter.calc_AP_AR(large_prec,large_rec)
    
    # print('Total')
    # print(ap,ar)
    # print('Small objs.')
    # print(aps,ars)
    # print('Medium objs.')
    # print(apm,arm)
    # print('Large objs.')
    # print(apl,arl)

    # print('Mean average precision over thresholds')
    # values = meter.average_map_mar_iou_threshold(pred)
    # print('Total')
    # print('Map: '+str(values[0]))
    # print('Mar: '+str(values[1]))
    # print('Small')
    # print('Map: '+str(values[2]))
    # print('Mar: '+str(values[3]))
    # print('Medium')
    # print('Map: '+str(values[4]))
    # print('Mar: '+str(values[5]))
    # print('Large')
    # print('Map: '+str(values[6]))
    # print('Mar: '+str(values[7]))
