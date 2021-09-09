# Highlights target and classify as small large and medium

import cv2

image_1 = r'C:\Users\Melgani\Desktop\master_degree\datasets\custom\test\097.png'
image_2 = r'C:\Users\Melgani\Desktop\master_degree\datasets\custom\test\016.png'

annotations_path = r'C:\Users\Melgani\Desktop\master_degree\datasets\custom\test\_annotations.txt'

ground_truth = {}
with open(annotations_path) as file:
    for line in file.readlines():
        pieces = line.split(' ')
        ground_truth[pieces[0]]=[]
        for box in pieces[1:]:
            pieces_box = box.split(',')
            ground_truth[pieces[0]].append([int(pieces_box[0]),int(pieces_box[1]),int(pieces_box[2]),int(pieces_box[3]),int(pieces_box[4]),1])

# Printing image_1 
annots = ground_truth['016.png']
image_cv = cv2.imread(image_2)
h,w,_ = image_cv.shape
for box in annots:
    area = (box[2]-box[0])*(box[3]-box[1])
    if(area<0.001*(h*w)):
        color = (0,255,0)
        imgHeight, imgWidth, _ = image_cv.shape
        thick = int((imgHeight + imgWidth) // 900)
        left = box[0]
        top = box[1]
        right = box[2]
        bottom = box[3]
        cv2.rectangle(image_cv,(left, top), (right, bottom), color, thick)
    elif(area>0.01*(h*w)):
        color = (255,0,0)
        imgHeight, imgWidth, _ = image_cv.shape
        thick = int((imgHeight + imgWidth) // 900)
        left = box[0]
        top = box[1]
        right = box[2]
        bottom = box[3]
        cv2.rectangle(image_cv,(left, top), (right, bottom), color, thick)
    else:
        color = (0,0,255)
        imgHeight, imgWidth, _ = image_cv.shape
        thick = int((imgHeight + imgWidth) // 900)
        left = box[0]
        top = box[1]
        right = box[2]
        bottom = box[3]
        cv2.rectangle(image_cv,(left, top), (right, bottom), color, thick)

cv2.imwrite('drawn_02.jpg', image_cv)