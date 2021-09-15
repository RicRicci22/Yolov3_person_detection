from numpy.lib.function_base import average
from detection import Detector
from metrics import Metric
from models import Yolov4
from tool.utils import *
import matplotlib.pyplot as plt

resolutions = [416,512,608,704,800,896,992,1088]
anno_path = r'datasets\custom\test\_annotations.txt'
dataset_path = r'datasets\custom\test'
# Creating the model
model = Yolov4(yolov4conv137weight=None,n_classes=80,inference=True)
model.load_weights(r'C:\Users\Melgani\Desktop\master_degree\weight\yolov4.pth')
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
# Small
# fig5 = plt.figure(figsize=(10,5))
# ax5 = fig5.subplots()

fig6 = plt.figure(figsize=(10,5))
ax6 = fig6.subplots()
# Medium
# fig7 = plt.figure(figsize=(10,5))
# ax7 = fig7.subplots()

fig8 = plt.figure(figsize=(10,5))
ax8 = fig8.subplots()
# Large
# fig9 = plt.figure(figsize=(10,5))
# ax9 = fig9.subplots()

fig10 = plt.figure(figsize=(10,5))
ax10 = fig10.subplots()

iou_list = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
confidence_steps = [0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.20,0.15,0.10,0.05]

for resolution in resolutions:
    average_prec_list = []
    average_rec_list = []
    mean_f1_list = []
    small_mean_rec_list = []
    medium_mean_rec_list = []
    large_mean_rec_list = []
    print('Calculating for resolution '+str(resolution)+'x'+str(resolution))
    # Creating the detector
    yolov4_detector = Detector(model,True,resolution,resolution,dataset_path,keep_aspect_ratio=False)
    # Get the predictions with a low confidence 
    predictions_dict,fps = yolov4_detector.detect_in_images(0.01,False,False)
    print(fps)
    for index_iou in range(len(iou_list)):
        print('Calculating for iou threshold '+str(iou_list[index_iou]))
        values = metrica.calculate_precision_recall_f1_lists(predictions_dict,confidence_steps,iou_list[index_iou],plot_graph=False)
        average_prec = metrica.calc_AP(values[0],values[1])

        average_prec_list.append(np.round(average_prec,3))
        average_rec_list.append(np.round(np.mean(values[1]),3))
        # mean of f1 scores for different confidences, and current iou
        mean_f1_list.append(np.mean(values[2]))

        small_mean_rec_list.append(np.mean(values[3]))

        medium_mean_rec_list.append(np.mean(values[4]))

        large_mean_rec_list.append(np.mean(values[5]))
    
    #print(average_prec_list)
    #print(average_rec_list)

    # Plotting the average vs iou lists
    ax.plot(range(len(iou_list)),average_prec_list,label=str(resolution)+'x'+str(resolution))
    ax2.plot(range(len(iou_list)),average_rec_list,label=str(resolution)+'x'+str(resolution))
    # Small
    
    ax6.plot(range(len(iou_list)),small_mean_rec_list,label=str(resolution)+'x'+str(resolution))
    # Medium 
    
    ax8.plot(range(len(iou_list)),medium_mean_rec_list,label=str(resolution)+'x'+str(resolution))
    # Large
    
    ax10.plot(range(len(iou_list)),large_mean_rec_list,label=str(resolution)+'x'+str(resolution))


# TOTAL 
ax.set_xticks(np.arange(len(iou_list)))
ax.set_xticklabels(iou_list)
ax.set_title('Average precision vs IoU')
ax.set_xlabel('IoU threshold for detection')
ax.set_ylabel('Average precision')
ax.legend()
fig.tight_layout()

ax2.set_xticks(np.arange(len(iou_list)))
ax2.set_xticklabels(iou_list)
ax2.set_title('Average recall vs IoU')
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