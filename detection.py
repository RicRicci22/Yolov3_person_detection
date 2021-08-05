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

    def detect_in_images_pytorch(self,confidence,output_file=False):
        # Perform prediction using yolov4 pytorch implementation
        # Creating the model
        model = Yolov4(n_classes=self.num_classes)
        # Creating the file to save output
        if(output_file):
            detections = open(self.testset+r'\_predictions.txt','w')
        # Create output prediction dictionary
        predictions_dict = {}
        # Starting the loop
        for filename in os.listdir(self.testset):
            f = os.path.join(self.testset, filename)
            if(os.path.isfile(f)):
                # Is the file an image??
                if(filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))):
                    print('Processing image '+filename)
                    img = cv2.imread(os.path.join(self.testset,filename))
                    original_height, original_width, _ = img.shape
                    sized = cv2.resize(img, (self.input_width, self.input_height))
                    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
                    #plt.imshow(sized)
                   # plt.show()
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



    def detect_in_images_darknet(self,confidence,output_file=False):
        # Perform prediction using darknet parser and yolov4.weights
        # This function will create a file in the same folder as the dataset containing the predictions in yolov4 pytorch format
        # INPUT
        # dataset_input = folder containing all the images where the detection has to be performed
        # output_file = if True, also output a file containing all the predictions in yolov4_pytorch format
        # OUTPUT
        # predictions_dict = dictionary containing predictions predictions_dict[filename] = [[box1],[box2],[box3]...]
        if(output_file):
            detections = open(self.testset+r'\_predictions.txt','w')
        # Create output prediction dictionary
        predictions_dict = {}
        for filename in os.listdir(self.testset):
            f = os.path.join(self.testset, filename)
            if(os.path.isfile(f)):
                # Is the file an image??
                if(filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))):
                    print('Processing image '+filename)
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
                # TODO label = ''
                cv2.rectangle(image_cv,(left, top), (right, bottom), color, thick)
                #cv2.putText(image_cv, label, (left, top - 12), 0, 1e-3 * imgHeight, color, thick//3)
            cv2.imwrite('predictions/drawn_'+image, image_cv)


def process_video_different_confidence(self, video_path, confidences):
    # Script that automatically process the same 2 videos using different confidence thresholds
    # confidences -> LIST of confidence values
    import fnmatch

    # Break path into pieces
    pieces = os.path.split(video_path)

    f = open(pieces[0]+"/average_FPS.txt", "a")
    f.write('\nFPS FOR DIFFERENT CONFIDENCE VALUES\n')

    for confidence in confidences:
        cap = cv2.VideoCapture(video_path)
        # Get info on video
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cap.set(3, 1280)
        cap.set(4, 720)
        print("Starting the YOLO loop...")
        # Loading class names
        class_names = load_class_names(self.namesfile)
        # Object to save the video
        width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer= cv2.VideoWriter(pieces[0]+'/result_'+str(confidence)+'_'+pieces[1], 0x00000021, fps, (width,height))
        n_frame = 1
        start = time.time()
        while (n_frame<=frame_count):
            print('Processing frame',n_frame,'out of',frame_count)
            ret, img = cap.read()
            sized = cv2.resize(img, (m.width, m.height))
            sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

            boxes = do_detect(self.model, sized, confidence, self.num_classes, 0.4, self.use_cuda)

            result_img = plot_boxes_cv2(img, boxes, savename=None, class_names=class_names)

            writer.write(result_img)
            cv2.waitKey(1)
            n_frame=n_frame+1
        finish = time.time()
        # Printing average FPS in a txt file inside the folder
        # TODO
        f.write(pieces[1]+'_confidence_'+str(confidence)+'-->'+str(round(frame_count/(finish-start),1))+'FPS\n')
        f.close
        cap.release()
        writer.release()
        # Compress video
        print('Compressing video to 360p..')
        compress_video(pieces[0]+'//result_'+str(confidence)+'_'+pieces[1])




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
    # # Creating the detector
    # yolov4_1024_1024 = Detector(model,True,1,736,736,r'C:\Users\farid.melgani\Desktop\master_degree\visdrone\test')
    # predictions = yolov4_1024_1024.detect_in_images_pytorch(0.4)
    # yolov4_1024_1024.visualize_predictions(predictions)

    # metrica = metrics.Metric(r'C:\Users\farid.melgani\Desktop\master_degree\visdrone\test\annotations\_annotations.txt',ground_truth_dict,yolov4_1024_1024)
    # precision,recall,f1 = metrica.precision_recall(predictions,0.5)
    # print(precision,recall,f1)
    # #metrica.plot_precision_recall_curve(0.10)

    # # DARKNET
    # # Creating the model
    model = Darknet('cfg/yolov4.cfg')
    model.load_weights(r'weight/yolov4.weights')
    model.activate_gpu()
    # # Creating the detector
    yolov4_1024_1024 = Detector(model,True,80,512,512,r'C:\Users\farid.melgani\Desktop\master_degree\visdrone\test')
    predictions = yolov4_1024_1024.detect_in_images_darknet(0.4)
    # yolov4_1024_1024.visualize_predictions(predictions)

    metrica = metrics.Metric(r'C:\Users\farid.melgani\Desktop\master_degree\visdrone\test\annotations\_annotations.txt',ground_truth_dict,yolov4_1024_1024)
    precision,recall,f1 = metrica.precision_recall(predictions,0.5)
    print(precision,recall,f1)
    # #metrica.plot_precision_recall_curve(0.10)