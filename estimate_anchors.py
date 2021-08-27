# Estimate the anchor boxes dimensions from training data

#import torch

#dictionary = torch.load(r'C:\Users\Melgani\Desktop\master_degree\weight\trained_weights\yolov4_trained_4_epochs_visdrone.pth')
#new_dict = {}
#for key, value in dictionary.items():
#    if ('neek' in key):
#        new_key = key.replace('neek','neck')
#        new_dict[new_key] = value
#    else:
#        new_dict[key] = value

#torch.save(new_dict,r'C:\Users\Melgani\Desktop\master_degree\weight\trained_weights\new_yolov4_trained_4_epochs_visdrone.pth')

# IMPORTS 
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

annotations_path = r'C:\Users\Riccardo\Desktop\TESI MAGISTRALE\Code\VISDRONE\visDrone_pedestrian_people_all_bboxes\train\train\_annotations.txt'
areas = []
ratios = []
widths = []
heights = []
with open(annotations_path,'r') as file:
    images = file.readlines()
    for image in images:
        bboxes = image.split(' ')[1:]
        for box in bboxes:
            coords = box.split(',')[:4]
            width = int(coords[2])-int(coords[0])
            height = int(coords[3])-int(coords[1])
            if(width*height!=0):
                areas.append(width*height)
                ratios.append(width/height)
                widths.append(width)
                heights.append(height)

x=[widths,heights]
x=np.asarray(x)
x=x.transpose()

kmeans3 = KMeans(n_clusters=9)
kmeans3.fit(x)
y_kmeans3 = kmeans3.predict(x)

centers3 = kmeans3.cluster_centers_

yolo_anchor_average=[]
for ind in range (9):
    yolo_anchor_average.append(np.mean(x[y_kmeans3==ind],axis=0))

yolo_anchor_average=np.array(yolo_anchor_average)

plt.scatter(x[:, 0], x[:, 1], c=y_kmeans3, s=2, cmap='viridis')
plt.scatter(yolo_anchor_average[:, 0], yolo_anchor_average[:, 1], c='red', s=50);
yoloV3anchors = yolo_anchor_average
yoloV3anchors[:, 0] =yolo_anchor_average[:, 0] /1280 *608
yoloV3anchors[:, 1] =yolo_anchor_average[:, 1] /720 *608
yoloV3anchors = np.rint(yoloV3anchors)
fig, ax = plt.subplots()
for ind in range(9):
    rectangle= plt.Rectangle((304-yoloV3anchors[ind,0]/2,304-yoloV3anchors[ind,1]/2), yoloV3anchors[ind,0],yoloV3anchors[ind,1] , fc='b',edgecolor='b',fill = None)
    ax.add_patch(rectangle)
ax.set_aspect(1.0)
plt.axis([0,608,0,608])
plt.show()
yoloV3anchors.sort(axis=0)
print("Your custom anchor boxes are {}".format(yoloV3anchors))

#plt.scatter(areas,ratios,s=1)
# plt.scatter(widths,heights,s=1)
# plt.title('scatterplot area ratio')
# plt.show()