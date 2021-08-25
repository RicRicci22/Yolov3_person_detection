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
import matplotlib.pyplot as plt

annotations_path = r'C:\Users\Melgani\Desktop\master_degree\datasets\visdrone\train\_annotations.txt'
areas = []
ratios = []
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

plt.scatter(areas,ratios,s=1)
plt.title('scatterplot area ratio')
plt.show()