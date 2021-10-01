# This file will contain functions to perform detection on various files such as images, videos, etc.

from metrics import Metric
from tool.utils import *
from tool.torch_utils import *
from models import Yolov4
import cv2
import os

class Detector:
    def __init__(self,model,use_cuda,input_width,input_height,testset,keep_aspect_ratio):
        self.model = model
        self.use_cuda = use_cuda
        self.testset = testset # Path to the folder containing the testset over which to perform detections
        self.input_width = input_width
        self.input_height = input_height
        # cfgfile can be None cause it's used only when using Darknet configuration file!
        # Find namesfile
        self.namesfile = 'data/custom_names.names'
        self.keep_aspect_ratio = keep_aspect_ratio
    
    def __str__(self):
        print('Num. classes: 1')
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
            # Get info on video3
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            print("Starting the YOLO loop...")

            width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            rgb = (0,0,255)
            writer= cv2.VideoWriter(video_folder+'/result '+video, 0x00000021, fps, (width,height))
            n_frame = 1
            while (n_frame<=frame_count):
                print('Processing frame',n_frame,'out of',frame_count)
                _, img = cap.read()
                sized = cv2.resize(img, (self.input_width, self.input_height))
                sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

                boxes,totaltime = do_detect(self.model, sized, confidence, 0.4, self.use_cuda)
                #print(boxes)
                new_boxes = [box for box in boxes[0] if box[5]==0]
                for box in new_boxes:
                    x1 = int(box[0]*width)
                    y1 = int(box[1]*height)
                    x2 = int(box[2]*width)
                    y2 = int(box[3]*height)
                    # check if negative
                    if(x1<0):
                        x1=0
                    if(y1<0):
                        y1=0
                    if(x2>width):
                        x2=width
                    if(y2>height):
                        y2=height
                    img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 3)
               
                #result_img = plot_boxes_cv2(img, boxes, savename=None, class_names=class_names)

                writer.write(img)
                cv2.waitKey(1)
                n_frame=n_frame+1
            cap.release()
            writer.release()

    def detect_in_images(self, confidence, output_file=False, visualize_predictions=False,ground_truth=[]):
        # Perform prediction using yolov4 pytorch implementation
        # Creating the file to save output
        if(output_file):
            detections = open(self.testset+r'\_predictions.txt','w')
        # Create output prediction dictionary
        predictions_dict = {}
        tot_img = 0
        tot_time = 0
        # Starting the loop
        for filename in os.listdir(self.testset):
            f = os.path.join(self.testset, filename)
            if(os.path.isfile(f)):
                # Is the file an image??
                if(filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))):
                    img = cv2.imread(os.path.join(self.testset,filename))
                    original_height, original_width, _ = img.shape
                    if(self.keep_aspect_ratio):
                        sized = keep_ratio(img,self.input_height)
                        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
                    else:
                        sized = cv2.resize(img, (self.input_width, self.input_height))
                        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
                    # perform detection
                    boxes, total_time = do_detect(self.model, sized, confidence, 0.2, self.use_cuda,print_time=False)
                    if(tot_img!=0):
                        tot_time += total_time
                    
                    tot_img += 1

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
        
        fps = (tot_img-1)/tot_time

        # Visualizing predictions 
        if(visualize_predictions):
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

                boxes_predicted = predictions_dict[image]
                #boxes_true = ground_truth[image]
                for box in boxes_predicted:
                    left = box[0]
                    top = box[1]
                    right = box[2]
                    bottom = box[3]
                    cv2.rectangle(image_cv,(left, top), (right, bottom), color, thick)
                # for box in boxes_true:
                #     left = box[0]
                #     top = box[1]
                #     right = box[2]
                #     bottom = box[3]
                #     cv2.rectangle(image_cv,(left, top), (right, bottom), (255,0,0), thick)
                cv2.imwrite('predictions/drawn_'+image, image_cv)

        return predictions_dict, fps


if __name__ == '__main__':
    # PYTORCH
    # Creating the model
    model = Yolov4(yolov4conv137weight=None,n_classes=80,inference=True)
    #model.load_weights(r'C:\Users\Melgani\Desktop\master_degree\trained_weights\all_datasets\finetuned_custom\100_epochs\custom800.pth')
    model.load_weights(r'C:\Users\Melgani\Desktop\master_degree\weight\yolov4.pth')
    model.activate_gpu()
    yolov4_detector = Detector(model,True,800,800,r'C:\Users\Melgani\Desktop\master_degree\datasets\positive_negative',keep_aspect_ratio=False)
    
    #yolov4_detector.process_all_videos(r'C:\Users\Melgani\Desktop\master_degree\Video material',0.5)
    # pred, fps = yolov4_detector.detect_in_images(0.5,False,True,metric_obj.ground_truth)
    # print(fps)
    # model_eval = Yolov4(yolov4conv137weight=None,n_classes=1,inference=True)
    # device = torch.device('cuda')
    # model_eval.load_weights(r'C:\Users\Melgani\Desktop\master_degree\trained_weights\Yolov4_epoch80.pth')
    # model_eval.to(device=device)
    # detector = Detector(model_eval,True,800,800,r'datasets\custom\test',keep_aspect_ratio=False)
    metric_obj = Metric(r'C:\Users\Melgani\Desktop\master_degree\datasets\positive_negative\_annotations.txt',r'C:\Users\Melgani\Desktop\master_degree\datasets\positive_negative')

    #pred,fps = yolov4_detector.detect_in_images(0.01)
    pred, fps = yolov4_detector.detect_in_images(0.5,False,True,metric_obj.ground_truth)
    print(fps)
    #print(pred)
    # confidence_steps = [0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1]
    # values = metric_obj.calculate_precision_recall_f1_lists(pred,confidence_steps,0.2)
    # print(values[0])
    # print(values[1])
    # # AP calc.
    # average_prec = metric_obj.calc_AP(values[0],values[1])
    # print(average_prec)

    metric_obj.frame_metric(pred,0.5)
