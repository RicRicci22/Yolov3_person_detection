import os
import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET


def format_annotations_visdrone_to_yolov4_pytorch(annotations_path,area_threshold=0):
    # This function will format the visdrone dataset to a yolov4 annotation
    # INPUT
    # annotations_path = path with the "old" annotations
    # annotations_formatted_path = path were the formatted annotations will be stored
    # area_threshold = if different from 0, select only bounding boxes of area > area_threshold. Area is expressed in number of pixels
    files = os.listdir(annotations_path)
    tot_bbox=0
    f_formatted = open(os.path.join(annotations_path,'_annotations.txt'),'w')
    for file in files:
        f = open(os.path.join(annotations_path, file),'r')
        control=0
        control_2 = 0
        for x in f.readlines():
            pieces = x.split(',')
            area = int(pieces[2])*int(pieces[3])
            if((int(pieces[5])==1 or int(pieces[5])==2) and area>area_threshold):
                # This is a person
                control = control+1
                control_2 = 1
            else:
                control_2 = 0
            if(control==1):
                name = file.split('.')
                f_formatted.write(name[0]+'.jpg')
                control = control+1
            if(control_2==1):
                f_formatted.write(' '+pieces[0]+','+pieces[1]+','+str(int(pieces[0])+int(pieces[2]))+','+str(int(pieces[1])+int(pieces[3]))+','+str(0))
                tot_bbox = tot_bbox+1
        if(control!=0):
            # Print the line feed
            f_formatted.write('\n')
        f.close()
    f_formatted.close()
    print('Total number of bboxes: ' + str(tot_bbox))


# This function will eliminate images that do not correspond to any annotation (remember we kept only person)
def remove_not_corresponding(images_path,annotations_path):
    # INPUT
    # images_path = path of the folder containing all the images
    # annotations_path = path to the file containing the annotations (the file is expected in yolov4_pytorch format)
    images = os.listdir(images_path)
    f = open(annotations_path,'r')
    rows = f.readlines()
    list_names = [row.split(' ')[0] for row in rows]
    i=0
    for image in images:
        i=i+1
        print("Processing image",i)
        if(not image in list_names):
            # Image should be deleted
            os.remove(os.path.join(images_path,image))

def analyze_dataset(annotations):
    bbox_areas = []
    # This function performs an analysis on the bounding box dimensions
    with open(annotations) as f:
        lines = f.readlines()

    for line in lines:
        # Every line contains bounding boxes for an image
        boxes = line.split(' ')
        for box in boxes[1:]:
            coords = box.split(',')
            width = int(coords[2])-int(coords[0])
            height = int(coords[3]) - int(coords[1])
            bbox_areas.append(width*height)

    # Get quantiles
    bbox_areas.sort()
    first_quantile = bbox_areas[round(len(bbox_areas)*0.33)]
    second_quantile = bbox_areas[round(len(bbox_areas)*0.66)]
    print(first_quantile)
    print(second_quantile)

    # Printing graphically
    n= plt.hist(x=bbox_areas, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Area')
    plt.ylabel('Frequency')
    plt.title('Bbox area distribution')
    plt.text(23, 45, r'$\mu=15, b=3$')
    plt.xlim((-100, 4000))
    plt.show()

def format_annotations_SARD_to_yolov4_pytorch(annotations_path,annotations_formatted_path):
    formatted_annotations = open(annotations_formatted_path, 'a')
    for _, _, files in os.walk(annotations_path):
        for name in files:
            if name.endswith((".xml")):
                pieces_name = name.split('.')
                formatted_annotations.write(pieces_name[0]+'.jpg')
                tree = ET.parse(os.path.join(annotations_path,name))
                root = tree.getroot()
                for object in root.iter('bndbox'):
                    formatted_annotations.write(' ')
                    for attrib in object.iter():
                        if(attrib.tag=='bndbox'):
                            continue
                        if(attrib.tag!='ymax'):
                            formatted_annotations.write(attrib.text+',')
                        else:
                            formatted_annotations.write(attrib.text)
                    formatted_annotations.write(',0')
                formatted_annotations.write('\n')
    formatted_annotations.close()

#def format_annotations_SARD_to_yolov4_pytorch(annotations_path):

