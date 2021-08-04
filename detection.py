# This file will contain functions to perform detection on various files such as images, videos, etc.

from tool.utils import *
from tool.darknet2pytorch import Darknet
from models import Yolov4
import argparse
import metrics
import format_dataset
import matplotlib.pyplot as plt
import cv2

class Detector:
    def __init__(self,use_cuda,num_classes,input_width,input_height,weightfile,namesfile,testset,cfgfile=None):
        self.use_cuda = use_cuda
        self.num_classes = num_classes
        self.cfgfile = cfgfile
        self.weightfile = weightfile
        self.testset = testset # Path to the folder containing the testset over which to perform detections
        self.namesfile = namesfile
        self.input_width = input_width
        self.input_height = input_height
        # cfgfile can be None cause it's used only when using Darknet configuration file!

        if num_classes == 20:
            self.namesfile = 'data/voc.names'
        elif num_classes == 80:
            self.namesfile = 'data/coco.names'
        else:
            self.namesfile = 'data/x.names'

    def __str__(self):
        print('Detector object')
        print('Num. classes: ',self.num_classes)
        print('Testset path: ',self.testset)
        print('Weightfile path: ',self.weightfile)
        if(self.cfgfile):
            print('Cfgfile path: ',self.cfgfile)

        return 'Cuda acceleration?: '+str(self.use_cuda)


    def process_all_videos(self, video_folder):
        import os
        import fnmatch

        m = Darknet(self.cfgfile)

        #m.print_network()
        m.load_weights(weightfile)
        print('Loading weights from %s... Done!' % (self.weightfile))

        if self.use_cuda:
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
            # TODO
            f = open(video_folder+"/average_FPS.txt", "a")
            f.write(video+'-->'+str(round(frame_count/(finish-start),1))+'FPS\n')
            f.close
            cap.release()
            writer.release()

    def detect_in_images_pytorch(self,confidence,output_file=False):
        # Perform prediction using yolov4 pytorch implementation and tolov4.conv.137.pth
        # Creating the model
        model = Yolov4(yolov4conv137weight=self.weightfile,n_classes=self.num_classes)
        # Activating gpu
        if (self.use_cuda):
            model.cuda()
            print('Cuda ENABLED')
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
                    # perform detection
                    boxes = do_detect(model, sized, confidence, self.num_classes, 0.4, self.use_cuda)
                    # Process boxes to keep only people
                    new_boxes = [box for box in boxes if box[6]==0]
                    if(len(new_boxes)!=0):
                        if(output_file):
                            detections.write(filename)
                        predictions_dict[filename] = []
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
                            # Insert prediction
                            if(output_file):
                                detections.write(' '+str(x1)+','+str(y1)+','+str(x2)+','+str(y2)+','+str(obj_class))
                            predictions_dict[filename].append([x1,y1,x2,y2,obj_class])
                    if(output_file):
                        if(len(new_boxes)!=0):
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

        # Initializing network
        m = Darknet(self.cfgfile)
        m.load_weights(self.weightfile)

        if self.use_cuda:
            m.cuda()
            print('GPU activated!')

        for filename in os.listdir(self.testset):
            f = os.path.join(self.testset, filename)
            if(os.path.isfile(f)):
                # Is the file an image??
                if(filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))):
                    print('Processing image '+filename)
                    img = cv2.imread(os.path.join(self.testset,filename))
                    original_height, original_width, _ = img.shape
                    sized = cv2.resize(img, (m.width, m.height))
                    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
                    # perform detection
                    boxes = do_detect(m, sized, confidence, self.num_classes, 0.4, self.use_cuda)
                    # Process boxes to keep only people
                    new_boxes = [box for box in boxes if box[6]==0]
                    if(len(new_boxes)!=0):
                        if(output_file):
                            detections.write(filename)
                        predictions_dict[filename] = []
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
                            # Insert prediction
                            if(output_file):
                                detections.write(' '+str(x1)+','+str(y1)+','+str(x2)+','+str(y2)+','+str(obj_class))
                            predictions_dict[filename].append([x1,y1,x2,y2,obj_class])
                    if(output_file):
                        if(len(new_boxes)!=0):
                            detections.write('\n')

        return predictions_dict


def process_video_different_confidence(self, video_path, confidences):
    # Script that automatically process the same 2 videos using different confidence thresholds
    # confidences -> LIST of confidence values
    import cv2
    import fnmatch

    # Initializing network
    print('Network initialization')
    m = Darknet(self.cfgfile)
    m.load_weights(self.weightfile)
    print('Loading weights from %s... Done!' % (self.weightfile))
    if self.use_cuda:
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
        # Loading class names
        class_names = load_class_names(self.namesfile)
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

            boxes = do_detect(m, sized, threshold, self.num_classes, 0.4, self.use_cuda)

            result_img = plot_boxes_cv2(img, boxes, savename=None, class_names=class_names)

            writer.write(result_img)
            cv2.waitKey(1)
            n_frame=n_frame+1
        finish = time.time()
        # Printing average FPS in a txt file inside the folder
        # TODO
        f.write(pieces[1]+'_confidence_'+str(threshold)+'-->'+str(round(frame_count/(finish-start),1))+'FPS\n')
        f.close
        cap.release()
        writer.release()
        # Compress video
        print('Compressing video to 360p..')
        compress_video(pieces[0]+'//result_'+str(threshold)+'_'+pieces[1])



if __name__ == '__main__':
    # Parsing ground truth
    ground_truth_dict = {}
    file = open(r'C:\Users\farid.melgani\Desktop\master_degree\visdrone\test\annotations\_annotations.txt')
    for row in file.readlines():
        pieces = row.split(' ')
        ground_truth_dict[pieces[0]]=[]
        for bbox in pieces[1:]:
            coords = bbox.split(',')
            ground_truth_dict[pieces[0]].append([int(coords[0]),int(coords[1]),int(coords[2]),int(coords[3]),int(coords[4])])

    #oggetto = Detector(True, 80,r'weight/yolov4.conv.137.pth',r'data/coco.names',r'C:\Users\farid.melgani\Desktop\master_degree\visdrone\test')
    oggetto_pytorch = Detector(True,80,608,608,r'weight/yolov4.conv.137.pth',r'data/coco.names',r'C:\Users\farid.melgani\Desktop\master_degree\visdrone\test',r'cfg/yolov4.cfg')
    oggetto_pytorch.detect_in_images_pytorch(0.5,False)
    #metrica = metrics.Metric(r'C:\Users\farid.melgani\Desktop\master_degree\visdrone\test\annotations\_annotations.txt',ground_truth_dict,oggetto)
    #metrica.plot_precision_recall_curve(0.25)