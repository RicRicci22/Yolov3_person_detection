from detection import Detector
from metrics import Metric
from models import Yolov4
from tool.utils import *
import pickle


# This module is meant to systematically process tests on dataset
# RESOLUTION TEST
# This test is meant to inspect how the input resolution influeces the detection performance
# TEST 1
# Parameters:
# Pretrained_weights
resolutions = [416,512,608,704,800,896,992,1088]
anno_path = r'datasets\visdrone\test\_annotations.txt'
dataset_path = r'datasets\visdrone\test'
# Creating the model
model = Yolov4(yolov4conv137weight=None,n_classes=80,inference=True)
model.load_weights(r'C:\Users\Melgani\Desktop\master_degree\weight\yolov4.pth')
model.activate_gpu()
# Creating element to measure metrics
metrica = Metric(anno_path,dataset_path)
# Creating file to store results
file = open(r'tests\input_resolution\visdrone\pretrained\no_keep_aspect_ratio\test.txt','w')
iou_list = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
confidence_steps = [0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.20,0.15,0.10,0.05]
for resolution in resolutions:
    print('Calculating for resolution '+str(resolution)+'x'+str(resolution))
    # Creating the detector
    yolov4_detector = Detector(model,True,resolution,resolution,dataset_path,keep_aspect_ratio=False)
    # Get the predictions with a low confidence 
    predictions_dict = yolov4_detector.detect_in_images(0.01,False,False)
    for iou in iou_list:
        print('Calculating for iou threshold '+str(iou))
        values = metrica.calculate_precision_recall_f1_curve(predictions_dict,confidence_steps,iou,plot_graph=False)
        # # Saving values
        # for i in range(len(confidence_steps)):
        #     file.write(str(resolution)+'x'+str(resolution)+'\n')
        #     file.write('Iou threshold for true positive: '+str(iou)+'\n')
        #     file.write('Confidence for prediction: '+str(confidence_steps[i])+'\n')
        #     file.write('Total precision: '+str(np.around(values[0][i],2))+'  Total recall: '+str(np.around(values[1][i],2))+'  Total f1 score: '+str(np.around(values[2][i]))+'\n')
        #     file.write('Small objects\n')
        #     file.write('Precision: '+str(np.around(values[3][i]))+'  Recall: '+str(np.around(values[4][i]))+'\n')
        #     file.write('Medium objects\n')
        #     file.write('Precision: '+str(np.around(values[5][i]))+'  Recall: '+str(np.around(values[6][i]))+'\n')
        #     file.write('Large objects\n')
        #     file.write('Precision: '+str(np.around(values[7][i]))+'  Recall: '+str(np.around(values[8][i],2))+'\n\n')
        # Calculating average precision and recall
        file.write(str(resolution)+'x'+str(resolution)+'\n')
        file.write('Iou threshold for true positive: '+str(iou)+'\n')
        average_prec, average_rec = metrica.calc_AP_AR(values[0],values[1])
        file.write('Average precision: '+str(np.around(average_prec,2))+'\n')
        file.write('Average recall: '+str(np.around(average_rec,2))+'\n\n')

file.close()