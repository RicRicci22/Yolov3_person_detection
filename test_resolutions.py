from detection import Detector
from metrics import Metric
from models import Yolov4
from tool.utils import *
import matplotlib.pyplot as plt

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

mean_average_prec_list = []
mean_average_rec_list = []

# Preparing graphs  
# Average prec vs iou 
fig = plt.figure(figsize=(10,5))
ax = fig.subplots()
# Average rec vs iou 
fig2 = plt.figure(figsize=(10,5))
ax2 = fig2.subplots()

# Mean average prec vs resolution 
fig3 = plt.figure(figsize=(10,5))
ax3 = fig3.subplots()
# Mean average rec vs resolution 
fig4 = plt.figure(figsize=(10,5))
ax4 = fig4.subplots()

for resolution in resolutions:
    average_prec_list = []
    average_rec_list = []
    print('Calculating for resolution '+str(resolution)+'x'+str(resolution))
    # Creating the detector
    yolov4_detector = Detector(model,True,resolution,resolution,dataset_path,keep_aspect_ratio=False)
    # Get the predictions with a low confidence 
    predictions_dict,fps = yolov4_detector.detect_in_images(0.01,False,False)
    print(fps)
    for index_iou in range(len(iou_list)):
        print('Calculating for iou threshold '+str(iou_list[index_iou]))
        values = metrica.calculate_precision_recall_f1_curve(predictions_dict,confidence_steps,iou_list[index_iou],plot_graph=False)
        average_prec, average_rec = metrica.calc_AP_AR(values[0],values[1])
        average_prec_list.append(average_prec)
        average_rec_list.append(average_rec)
    mean_average_prec_list.append(sum(average_prec_list)/len(average_prec_list))
    mean_average_rec_list.append(sum(average_rec_list)/len(average_rec_list))

    # Plotting the average vs iou lists
    ax.plot(range(len(iou_list)),average_prec_list,label=str(resolution)+'x'+str(resolution))
    ax2.plot(range(len(iou_list)),average_rec_list,label=str(resolution)+'x'+str(resolution))

# Plotting the mean average vs resolution list
ax3.plot(range(len(resolutions)),mean_average_prec_list)
ax4.plot(range(len(resolutions)),mean_average_rec_list)


ax.set_xticklabels(iou_list)
ax.set_xlabel('IoU threshold for detection')
ax.set_ylabel('Average precision')
ax.set_title('AVERAGE PRECISION VS IOU')
ax.legend()
fig.tight_layout()

ax2.set_xticklabels(iou_list)
ax2.set_xlabel('IoU threshold for detection')
ax2.set_ylabel('Average recall')
ax2.set_title('AVERAGE RECALL VS IOU')
ax2.legend()
fig2.tight_layout()

ax3.set_xticklabels(resolutions)
ax3.set_xlabel('Input resolution')
ax3.set_ylabel('Mean average precision over IoU threshold')
ax3.set_title('MEAN AVERAGE PRECISION VS RESOLUTION')
ax3.legend()
fig3.tight_layout()

ax4.set_xticklabels(iou_list)
ax4.set_xlabel('Input resolution')
ax4.set_ylabel('Mean average recall over IoU threshold')
ax4.set_title('AVERAGE RECALL VS RESOLUTION')
ax4.legend()
fig4.tight_layout()


plt.show()