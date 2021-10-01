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

resolutions = [800]
anno_path = r'datasets\custom\test\_annotations.txt'
dataset_path = r'datasets\custom\test'

model = Yolov4(yolov4conv137weight=None,n_classes=1,inference=True,anchors=custom_anchors)
model.load_weights(r'C:\Users\Melgani\Desktop\master_degree\trained_weights\all_datasets\finetuned_custom\100_epochs\custom800.pth')
model.activate_gpu()

# Creating element to measure metrics
metrica = Metric(anno_path,dataset_path)

fig = plt.figure(figsize=(10,5))
ax = fig.subplots()

#confidence_steps = [0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.20,0.15,0.10,0.05]
confidence_steps = [i/100 for i in range(99,1,-1)]

# Creating the detector
yolov4_detector = Detector(model,True,800,800,dataset_path,keep_aspect_ratio=False)
# Get the predictions with a low confidence 
predictions_dict,fps = yolov4_detector.detect_in_images(0.01,False,False)

values = metrica.calculate_precision_recall_f1_lists(predictions_dict,confidence_steps,0.5,plot_graph=False)
values2 = metrica.calculate_precision_recall_f1_lists(predictions_dict,confidence_steps,0.75,plot_graph=False)

ax.plot(values[1],values[0],label='Lol')
ax.plot(values2[0],values2[1],label='asd')
# ax.set_xlim(0.0,1.0)
# ax.set_ylim(0.0,1.0)

plt.show()