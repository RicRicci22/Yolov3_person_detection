from numpy.lib.function_base import average
from detection import Detector
from metrics import Metric
from models import Yolov4
from tool.utils import *
import matplotlib.pyplot as plt

original_anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
visdrone_anchors=[7, 11, 12, 21, 16, 32, 22, 35, 33, 48, 36, 67, 48, 71, 79, 100, 94, 168]
sard_anchors = [21,34,29,39,49,60,52,61,71,78,91,110,97,125,151,160,290,173]
custom_anchors = [14,22, 25,50, 39,57, 51,75, 52,76, 63,105, 72,125, 112,268, 135,306]

resolutions = [416,608,800,1088]
#resolutions = [800]
anno_path = r'datasets\visdrone\test\_annotations.txt'
dataset_path = r'datasets\visdrone\test'
# Creating the model
original_yolo = Yolov4(yolov4conv137weight=None,n_classes=80,inference=True,anchors=original_anchors)
original_yolo.load_weights(r'C:\Users\Melgani\Desktop\master_degree\weight\yolov4.pth')
original_yolo.activate_gpu()
model = Yolov4(yolov4conv137weight=None,n_classes=1,inference=True,anchors=sard_anchors)
model.load_weights(r'C:\Users\Melgani\Desktop\master_degree\trained_weights\sard\sard800.pth')
model.activate_gpu()
# Creating element to measure metrics
metrica = Metric(anno_path,dataset_path)

# Preparing graphs  
# Average prec vs iou 
fig = plt.figure(figsize=(10,5))
ax = fig.subplots()
# Average rec vs iou 
fig2 = plt.figure(figsize=(10,5))
ax2 = fig2.subplots()
fig6 = plt.figure(figsize=(10,5))
ax6 = fig6.subplots()
fig8 = plt.figure(figsize=(10,5))
ax8 = fig8.subplots()
fig10 = plt.figure(figsize=(10,5))
ax10 = fig10.subplots()

iou_list = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
confidence_steps = [i/100 for i in range(99,1,-1)]
#print(confidence_steps[24],confidence_steps[49],confidence_steps[74])

for resolution in resolutions:
    average_prec_list = []
    average_rec_list = []
    small_mean_rec_list = []
    medium_mean_rec_list = []
    large_mean_rec_list = []
    print('Calculating for resolution '+str(resolution)+'x'+str(resolution))
    # Creating the detector
    yolov4_detector = Detector(original_yolo,True,resolution,resolution,dataset_path,keep_aspect_ratio=False)

    predictions_dict,fps = yolov4_detector.detect_in_images(0.01,False,False)
    #indexes = []
    
    for index_iou in range(len(iou_list)):
        print('Calculating for iou threshold '+str(iou_list[index_iou]))
        values = metrica.calculate_precision_recall_f1_lists(predictions_dict,confidence_steps,iou_list[index_iou],plot_graph=False)
        average_prec = metrica.calc_AP(values[0],values[1])
        #print(iou_list[index_iou])
        #print('Average precision ',average_prec)

        #print(values[1][24],values[1][49],values[1][74])

        average_prec_list.append(np.round(average_prec,3))
        average_rec_list.append(np.round(np.mean([values[1][24],values[1][49],values[1][74]]),3))
        #print('Recall ',np.round(np.mean([values[1][24],values[1][49],values[1][74]]),3),'\n')

        small_mean_rec_list.append(np.mean([values[3][24],values[3][49],values[3][74]]))

        medium_mean_rec_list.append(np.mean([values[4][24],values[4][49],values[4][74]]))

        large_mean_rec_list.append(np.mean([values[5][24],values[5][49],values[5][74]]))
    
    # Plotting the average vs iou lists
    ax.plot(range(len(iou_list)),average_prec_list,label=str(resolution)+'x'+str(resolution))
    ax2.plot(range(len(iou_list)),average_rec_list,label=str(resolution)+'x'+str(resolution))
    # Small
    
    ax6.plot(range(len(iou_list)),small_mean_rec_list,label=str(resolution)+'x'+str(resolution))
    # Medium 
    
    ax8.plot(range(len(iou_list)),medium_mean_rec_list,label=str(resolution)+'x'+str(resolution))
    # Large
    
    ax10.plot(range(len(iou_list)),large_mean_rec_list,label=str(resolution)+'x'+str(resolution))
    
    # orig_average_prec_list = []
    # orig_average_rec_list = []
    # orig_small_mean_rec_list = []
    # orig_medium_mean_rec_list = []
    # orig_large_mean_rec_list = []
    # yolov4_detector = Detector(original_yolo,True,resolution,resolution,dataset_path,keep_aspect_ratio=False)
    # # Get the predictions with a low confidence 
    # predictions_dict,fps = yolov4_detector.detect_in_images(0.01,False,False)
    # for index_iou in range(len(iou_list)):
    #     print('Calculating for iou threshold '+str(iou_list[index_iou]))
    #     values = metrica.calculate_precision_recall_f1_lists(predictions_dict,confidence_steps,iou_list[index_iou],plot_graph=False)
    #     average_prec = metrica.calc_AP(values[0],values[1])
    #     print('Original')
    #     print(average_prec)

    #     orig_average_prec_list.append(np.round(average_prec,3))
    #     orig_average_rec_list.append(np.round(np.mean(values[1]),3))
    #     print('Recall ',np.round(np.mean(values[1]),3))

    #     orig_small_mean_rec_list.append(np.mean(values[3]))

    #     orig_medium_mean_rec_list.append(np.mean(values[4]))

    #     orig_large_mean_rec_list.append(np.mean(values[5]))
    
    # #print(average_prec_list)
    # #print(average_rec_list)

    # # Plotting the average vs iou lists
    # ax.plot(range(len(iou_list)),orig_average_prec_list,label='Original model')
    # ax2.plot(range(len(iou_list)),orig_average_rec_list,label='Original model')
    # # Small
    
    # ax6.plot(range(len(iou_list)),orig_small_mean_rec_list,label='Original model')
    # # Medium 
    
    # ax8.plot(range(len(iou_list)),orig_medium_mean_rec_list,label='Original model')
    # # Large
    
    # ax10.plot(range(len(iou_list)),orig_large_mean_rec_list,label='Original model')


# TOTAL 
ax.set_xticks(np.arange(len(iou_list)))
ax.set_xticklabels(iou_list)
#ax.set_title('Average precision vs IoU')
ax.set_xlabel('IoU threshold for detection')
ax.set_ylabel('Average precision')
ax.legend()
fig.tight_layout()

ax2.set_xticks(np.arange(len(iou_list)))
ax2.set_xticklabels(iou_list)
#ax2.set_title('Average recall vs IoU')
ax2.set_xlabel('IoU threshold for detection')
ax2.set_ylabel('Average recall')
ax2.legend()
fig2.tight_layout()

# Small 

ax6.set_xticks(np.arange(len(iou_list)))
ax6.set_xticklabels(iou_list)
ax6.set_title('Average recall -- Small objects')
ax6.set_xlabel('IoU threshold for detection')
ax6.set_ylabel('Average recall')
ax6.legend()
fig6.tight_layout()

# Medium

ax8.set_xticks(np.arange(len(iou_list)))
ax8.set_xticklabels(iou_list)
ax8.set_title('Average recall -- Medium objects')
ax8.set_xlabel('IoU threshold for detection')
ax8.set_ylabel('Average recall')
ax8.legend()
fig8.tight_layout()

# Large

ax10.set_xticks(np.arange(len(iou_list)))
ax10.set_xticklabels(iou_list)
ax10.set_title('Average recall -- Large objects')
ax10.set_xlabel('IoU threshold for detection')
ax10.set_ylabel('Average recall')
ax10.legend()
fig10.tight_layout()



plt.show()