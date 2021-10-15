from numpy.core.fromnumeric import argmax
from detection import Detector
from metrics import Metric
from models import Yolov4
from tool.utils import *
import matplotlib.pyplot as plt

original_anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
visdrone_anchors=[7, 11, 12, 21, 16, 32, 22, 35, 33, 48, 36, 67, 48, 71, 79, 100, 94, 168]
sard_anchors = [21,34,29,39,49,60,52,61,71,78,91,110,97,125,151,160,290,173]
custom_anchors = [14,22, 25,50, 39,57, 51,75, 52,76, 63,105, 72,125, 112,268, 135,306]

resolutions = [800]
anno_path = r'datasets\custom\test\_annotations.txt'
dataset_path = r'datasets\custom\test'

# model = Yolov4(yolov4conv137weight=None,n_classes=1,inference=True,anchors=sard_anchors)
# model.load_weights(r'C:\Users\Melgani\Desktop\master_degree\trained_weights\sard\sard800.pth')
# model.activate_gpu()
# Creating the model
original_yolo = Yolov4(yolov4conv137weight=None,n_classes=1,inference=True,anchors=custom_anchors)
original_yolo.load_weights(r'C:\Users\Melgani\Desktop\master_degree\trained_weights\custom800.pth')
original_yolo.activate_gpu()

original_yolo2 = Yolov4(yolov4conv137weight=None,n_classes=80,inference=True,anchors=original_anchors)
original_yolo2.load_weights(r'C:\Users\Melgani\Desktop\master_degree\weight\yolov4.pth')
original_yolo2.activate_gpu()

# Creating element to measure metrics
metrica = Metric(anno_path,dataset_path)

fig = plt.figure(figsize=(10,5))
ax = fig.subplots()

#confidence_steps = [0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.20,0.15,0.10,0.05]
confidence_steps = [i/100 for i in range(99,1,-1)]
# Creating the detector
#yolov4_detector = Detector(model,True,800,800,dataset_path,keep_aspect_ratio=False)
for resolution in resolutions:
    detector_2 = Detector(original_yolo,True,resolution,resolution,dataset_path,keep_aspect_ratio=False)
    #detector_3 = Detector(original_yolo2,True,800,800,dataset_path,keep_aspect_ratio=False)
    # Get the predictions with a low confidence 
    #predictions_dict,fps = yolov4_detector.detect_in_images(0.01,False,False)
    pred2,fps = detector_2.detect_in_images(0.01,False,False)
    #pred3,fps = detector_3.detect_in_images(0.01,False,False)

    #values = metrica.calculate_precision_recall_f1_lists(predictions_dict,confidence_steps,0.75,plot_graph=False)
    values2 = metrica.calculate_precision_recall_f1_lists(pred2,confidence_steps,0.5,plot_graph=False)
    #values3 = metrica.calculate_precision_recall_f1_lists(pred3,confidence_steps,0.75,plot_graph=False)

    # Inserting zeros and ones in lists 
    precisions = values2[0]
    recalls = values2[1]
    print('AP:',str(metrica.calc_AP(precisions,recalls)))
    print('Recall ',np.round(np.mean(recalls),3))
    ax.plot(values2[2],zorder=1)
    ax.scatter(argmax(values2[2]),max(values2[2]),color='Yellow',zorder=2,label='Confidence '+str(confidence_steps[argmax(values2[2])]))
    # precisions.insert(0,1)
    # precisions.append(0)
    # recalls.insert(0,0)
    # recalls.append(1)

    # precisions2 = values2[0]
    # #print(precisions2[0])
    # if(precisions2[0]==0):
    #     precisions2[0]=precisions2[2]
    #     precisions2[1]=precisions2[2]
    # recalls2 = values2[1]
    # plt.plot(recalls2,precisions2,zorder=1)
    # if(resolution==800):
    #     plt.scatter(recalls2[49],precisions2[49],color='Red',s=50,zorder=2,label='confidence=0.5')
    #     plt.scatter(recalls2[24],precisions2[24],color='Yellow',s=50,zorder=2,label='confidence=0.75')
    # else:
    #     plt.scatter(recalls2[49],precisions2[49],color='Red',s=50,zorder=2)
    #     plt.scatter(recalls2[24],precisions2[24],color='Yellow',s=50,zorder=2)
    #print('AP:',str(metrica.calc_AP(precisions2,recalls2)))
    #print('Recall ',np.round(np.mean(recalls2),3))

plt.xlabel('Confidence')
plt.ylabel('F1 score')
ax.set_xticks([i*3 for i in range(33)])
ax.set_xticklabels([i/100 for i in range(99,1,-3)],rotation=40)
plt.legend()
plt.show()
# precisions2.insert(0,1)
# precisions2.append(0)
# recalls2.insert(0,0)
# recalls2.append(1)

# precisions3 = values3[0]
# recalls3 = values3[1]
# print('AP:',str(metrica.calc_AP(precisions3,recalls3)))
# print('Recall ',np.round(np.mean(recalls3),3))
# precisions3.insert(0,1)
# precisions3.append(0)
# recalls3.insert(0,0)
# recalls3.append(1)
#print(precisions)
#print(recalls)

# ax.plot(recalls,precisions,label='100 epochs on sard')
# ax.plot(recalls2,precisions2,label='100 epochs on all datasets, 10 epochs on sard')
# ax.plot(recalls3,precisions3,label='100 epochs on all datasets, 100 epochs on sard')

# Calculate AP50 e AP75 


#ax.plot(values2[0],values2[1],label='asd')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.legend()
# plt.tight_layout()


# plt.show()