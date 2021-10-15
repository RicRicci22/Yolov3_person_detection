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
anno_path = r'C:\Users\Melgani\Desktop\master_degree\datasets\custom\test\_annotations.txt'
dataset_path = r'C:\Users\Melgani\Desktop\master_degree\datasets\custom\test'
model = Yolov4(yolov4conv137weight=None,n_classes=1,inference=True,anchors=custom_anchors)
model.load_weights(r'C:\Users\Melgani\Desktop\master_degree\trained_weights\all_datasets\finetuned_custom\100_epochs\custom800.pth')
model.activate_gpu()

metrica = Metric(anno_path,dataset_path)

yolov4_detector = Detector(model,True,800,800,dataset_path,keep_aspect_ratio=False)
predictions_dict,fps = yolov4_detector.detect_in_images(0.5,False,True)
#confidence_steps = [i/100 for i in range(99,1,-1)]
#values = metrica.calculate_precision_recall_f1_lists(predictions_dict,confidence_steps,0.5,plot_graph=False)

# average_prec = metrica.calc_AP(values[0],values[1])
# print(average_prec)
# print('Recall ',np.round(np.mean(values[1]),3))

metrica.frame_metric(predictions_dict,0.5)