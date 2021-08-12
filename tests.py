from detection import Detector
from metrics import Metric
from tool.darknet2pytorch import Darknet
from models import Yolov4
from tool.utils import *
import os
import pickle


# This module is meant to systematically process tests on dataset
# RESOLUTION TEST
# This test is meant to inspect how the input resolution influeces the detection performance
# TEST 1
# Parameters:
# Darknet with pretrained weights. Changing input resolution. Confidence threshold set to 0.5. IoU thrshold set to 0.5.
# Objective -> get precision and recall for each input resolution on the complete VISDRONE test set and write it on a txtfile. Also get precision, recall for small medium and large objects on the complete visdrone.
resolutions = [416,512,608,704,800,896,992,1088]
anno_path = r'C:\Users\farid.melgani\Desktop\master_degree\datasets\custom_dataset\test\_annotations.txt'
ground_truth_dict = parse_gtruth(anno_path)
# DARKNET
# Creating the model
model = Yolov4(yolov4conv137weight=None,n_classes=80,inference=True)
model.load_weights(r'C:\Users\farid.melgani\Desktop\master_degree\weight\yolov4.pth')
model.activate_gpu()
# Creating element to measure metrics
metrica = Metric(anno_path,ground_truth_dict)
# Creating file to store results
file = open(r'C:\Users\farid.melgani\Desktop\master_degree\tests\input_resolution\custom\pretrained_pytorch\test2.txt','w')
file.write('Pytorch yolov4 with pretrained weights. Changing input resolution. Confidence threshold set to 0.5. IoU thrshold set to 0.5. Small obj threshold is 72. Large obj threshold is 242. \n')
#############
# boxes = ground_truth_dict['0000006_00159_d_0000001.jpg']
# for box in boxes:
#     area = (box[3]-box[1])*(box[2]-box[0])
#     print(area)
#############
for resolution in resolutions:
    # Creating the detector
    print('Calculating for resolution '+str(resolution)+'x'+str(resolution))
    yolov4_detector = Detector(model,True,80,resolution,resolution,r'C:\Users\farid.melgani\Desktop\master_degree\datasets\custom_dataset\test')
    predictions = yolov4_detector.detect_in_images(0.4,True)
    filehandler = open(r'C:\Users\farid.melgani\Desktop\master_degree\tests\input_resolution\custom\pretrained_pytorch\predictions_'+str(resolution)+'.pkl', 'wb')
    pickle.dump(predictions, filehandler)
    overall_values = metrica.precision_recall(predictions,0.5)
    #small_values, medium_values, large_values = metrica.calculate_precision_recall_small_medium_large(predictions,0.5,72,242)
    # Saving values
    file.write(str(resolution)+'x'+str(resolution)+'\n')
    file.write('Overall precision: '+str(overall_values[0])+'  Overall recall: '+str(overall_values[1])+'  Overall f1 score: '+str(overall_values[2])+'\n')
    #file.write('Small obj. precision: '+str(small_values[0])+'  Small obj. recall: '+str(small_values[1])+'  Small obj. f1 score: '+str(small_values[2])+'\n')
    #file.write('Medium obj. precision: '+str(medium_values[0])+'  Medium obj. recall: '+str(medium_values[1])+'  Medium obj. f1 score: '+str(medium_values[2])+'\n')
    #file.write('Large obj. precision: '+str(large_values[0])+'  Large obj. recall: '+str(large_values[1])+'  Large obj. f1 score: '+str(large_values[2])+'\n\n')

file.close()