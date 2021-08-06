# This file will contain functions to perform detection on various files such as images, videos, etc.

from tool.utils import *
from tool.darknet2pytorch import Darknet
from models import Yolov4
import argparse
import metrics
import format_dataset
import matplotlib.pyplot as plt
import cv2
import os
import time

class Detector:
    def __init__(self,model,use_cuda,num_classes,input_width,input_height,testset):
        self.model = model
        self.use_cuda = use_cuda
        self.num_classes = num_classes
        self.testset = testset # Path to the folder containing the testset over which to perform detections
        self.input_width = input_width
        self.input_height = input_height
        # cfgfile can be None cause it's used only when using Darknet configuration file!
        # Find namesfile
        if num_classes == 20:
            self.namesfile = 'data/voc.names'
        elif num_classes == 80:
            self.namesfile = 'data/coco.names'
        else:
            self.namesfile = 'data/custom_names.names'

    def __str__(self):
        print('Detector object')
        print('Num. classes: ',self.num_classes)
        print('Testset path: ',self.testset)
        print('Model: ',self.model)
        if(self.cfgfile):
            print('Cfgfile path: ',self.cfgfile)

        return 'Cuda acceleration?: '+str(self.use_cuda)


    def process_all_videos(self, video_folder, confidence):
        import os
        import fnmatch

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
            class_names = load_class_names(self.namesfile)
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
                boxes = do_detect(self.model, sized, confidence, self.num_classes, 0.4, use_cuda)
                #finish = time.time()
                #print('Predicted in %f seconds.' % (finish - start))

                result_img = plot_boxes_cv2(img, boxes, savename=None, class_names=class_names)

                # cv2.imshow('Yolo demo', result_img)
                writer.write(result_img)
                cv2.waitKey(1)
                n_frame=n_frame+1
            finish = time.time()
            # Printing average FPS in a txt file inside the folder
            # TODO
            f = open(video_folder+"/average_FPS.txt", "a")
            f.write(video+'-->'+str(round(frame_count/(finish-start),1))+'FPS\n')
            f.close
            cap.release()
            writer.release()

    def detect_in_images(self,confidence,output_file=False):
        # Perform prediction using yolov4 pytorch implementation
        # Creating the model
        model = Yolov4(n_classes=self.num_classes)
        # Creating the file to save output
        if(output_file):
            detections = open(self.testset+r'\_predictions.txt','w')
        # Create output prediction dictionary
        predictions_dict = {}
        # Starting the loop
        print('Detecting in images..')
        for filename in os.listdir(self.testset):
            f = os.path.join(self.testset, filename)
            if(os.path.isfile(f)):
                # Is the file an image??
                if(filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))):
                    #print('Processing image '+filename)
                    img = cv2.imread(os.path.join(self.testset,filename))
                    original_height, original_width, _ = img.shape
                    sized = cv2.resize(img, (self.input_width, self.input_height))
                    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
                    # perform detection
                    boxes = do_detect(self.model, sized, confidence, self.num_classes, 0.4, self.use_cuda)
                    # Process boxes to keep only people
                    new_boxes = [box for box in boxes if box[6]==0]
                    if(output_file):
                        detections.write(filename)
                    predictions_dict[filename] = []
                    for box in new_boxes:
                        x1 = int((box[0] - box[2] / 2.0) * original_width)
                        y1 = int((box[1] - box[3] / 2.0) * original_height)
                        x2 = int((box[0] + box[2] / 2.0) * original_width)
                        y2 = int((box[1] + box[3] / 2.0) * original_height)
                        # check if negative
                        if(x1<0):
                            x1=0
                        if(y1<0):
                            y1=0
                        # TODO think about that
                        if(x2<0):
                            x2=0
                        if(y2<0):
                            y2=0
                        obj_class = box[6] #0 means person
                        # Insert prediction
                        if(output_file):
                            detections.write(' '+str(x1)+','+str(y1)+','+str(x2)+','+str(y2)+','+str(obj_class))
                        predictions_dict[filename].append([x1,y1,x2,y2,obj_class])
                    if(output_file):
                        detections.write('\n')

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

    def calculate_predictions_list(self, conf_step):
        # conf_step = the step to calculate the confidence levels
        conf_values = [i/100 for i in range(1,100,int(conf_step*100))]
        conf_values.append(0.99)
        predictions = []
        for conf in conf_values:
            print('Detecting in images with confidence: '+str(conf))
            predictions.append(self.detect_in_images(conf))

        return predictions


if __name__ == '__main__':
    # Parsing ground truth
    ground_truth_dict = {}
    file = open(r'C:\Users\farid.melgani\Desktop\master_degree\visdrone\test\annotations\_annotations.txt','r')
    for row in file.readlines():
        pieces = row.split(' ')
        ground_truth_dict[pieces[0]]=[]
        for bbox in pieces[1:]:
            coords = bbox.split(',')
            ground_truth_dict[pieces[0]].append([int(coords[0]),int(coords[1]),int(coords[2]),int(coords[3]),int(coords[4])])

    # # PYTORCH
    # # Creating the model
    # model = Yolov4(1)
    # model.load_weights(r'trained_weights/Visdrone_person+pedestrian/Yolov4_epoch1.pth')
    # model.activate_gpu()

    # # DARKNET
    # # Creating the model
    model = Yolov4(1)
    model.load_weights(r'C:\Users\farid.melgani\Desktop\master_degree\trained_weights\Visdrone_person+pedestrian_area_mor_200\3_epochs_0.001lr_batch5.pth')
    model.activate_gpu()

    # Creating the detector
    yolov4_1024_1024 = Detector(model,True,1,512,512,r'C:\Users\farid.melgani\Desktop\master_degree\visdrone\test')
    predictions = yolov4_1024_1024.detect_in_images(0.4)
    prediction_list = yolov4_1024_1024.calculate_predictions_list(0.25)

    metrica = metrics.Metric(r'C:\Users\farid.melgani\Desktop\master_degree\visdrone\test\annotations\_annotations.txt',ground_truth_dict)
    normal_values = metrica.precision_recall(predictions,0.5)
    small_values, medium_values, large_values = metrica.calculate_precision_recall_small_medium_large(predictions,0.5)
    print(normal_values[0],normal_values[1],normal_values[2])
    print(small_values[0],small_values[1],small_values[2])
    print(medium_values[0],medium_values[1],medium_values[2])
    print(large_values[0],large_values[1],large_values[2])
    precision_list, recall_list = metrica.calculate_precisio_recall_lists(prediction_list,0.5)
    print(metrica.calc_average_precision(precision_list,recall_list))