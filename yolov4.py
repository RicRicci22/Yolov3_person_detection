import cv2
import numpy as np

# All the functions concerning object detection using yolov4

def setup_network(input_width, input_heigth, scale_image):
    # Setup a yolo network
    
    # INPUTS:
    # input_width, input_heigth -> resolution of the input image (can be lower than the original image)
    # scale_image -> 

    # OUTPUTS: 
    #model -> yolov4 model ready to go 
    
    net =  cv2.dnn.readNet("model/yolov4.weights", "model/yolov4.cfg")
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(input_width, input_heigth), scale=scale_image, swapRB=True)
    return model

def detect_with_yolov4(model, input_image, conf_threshold, nms_threshold, desired_classes, save_detected_objects=False):
    # Function that detects object in an image

    # INPUTS:
    # model -> the yolov4 model to use 
    # input_image -> the image on which to perform detection
    # conf_threshold -> confidence threshold for the detection
    # nms_threshold -> non-maximum suppression threshold
    # desired classes -> LIST with all the name of the classes to detect (must be in supported class)
    # save_detected_objects -> whether or not the algorithm must save the cropped image on the detected target(s)

    # OUTPUTS:
    # input_image -> input image with annotated detections

    # Set up parameters
    LOOKING_FOR = []
    classNames = []
    with open("model/classList.txt", "r") as f:
        for name in f.readlines():
            name = name.strip()
            classNames.append(name)
            if name in desired_classes:
                LOOKING_FOR.append(len(classNames) - 1)
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')

    # Detect
    classes, scores, boxes = model.detect(input_image, conf_threshold, nms_threshold)

    # Save 
    objectCount=0
    for (id, score, box) in zip(classes, scores, boxes):
        if id[0] not in LOOKING_FOR:
            continue

        if save_detected_objects:
            objectCount += 1
            x, y, w, h = box
            objectROI = input_image[y:y+h, x:x+w]
            cv2.imwrite("detectedObjects/"+classNames[id[0]]+str(objectCount)+".jpg", objectROI)

        color = [int(c) for c in colors[id[0]]]
        cv2.rectangle(input_image, box, color, 2)
        cv2.putText(input_image, classNames[id[0]], (box[0], box[1] - 5), cv2.FONT_ITALIC, 0.8, (255, 255, 255), 2)

    return input_image

def detect_in_video_with_yolov4(model, video_path, video_fps, conf_threshold, nms_threshold, desired_classes, save_output_video, save_output_name="", skip_frames=False, skip_count=0, save_detected_objects=False):
    # This function runs yolov4 on an input video 
    
    # INPUTS:
    # model -> the yolov4 model to use 
    # video_path -> path to the video 
    # conf_threshold -> confidence threshold for the detection
    # nms_threshold -> non-maximum suppression threshold
    # desired classes -> LIST with all the name of the classes to detect (must be in supported class)
    # save_output_video -> whether or not to save the output video (will be saved in "video" folder)
    # save_output_name -> the name of the output video 
    # skip_frames -> boolean specifying if the algo has to skip some frames to speed up detection
    # skip_count -> number of frames to skip 
    # save_detected_objects -> whether or not the algorithm must save the cropped image on the detected target(s)

    # Setting up parameters
    skip = 0
    writer = 0
    vc = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    while cv2.waitKey(1) < 1:
        (grabbed, frame) = vc.read()
        if not grabbed:
            if save_output_video:
                writer.release()
            exit()

        if save_output_video and writer == 0:
            frameHeight, frameWidth, _ = frame.shape
            writer = cv2.VideoWriter('video/'+save_output_name+'.mp4', fourcc, video_fps,
                                    (frameWidth, frameHeight))

        if skip_frames:
            if skip >= skip_count:
                skip = 0
            else:
                skip += 1
                cv2.imshow("detections", frame)
                if save_output_video:
                    writer.write(frame)
                continue
        
        
        frame = detect_with_yolov4(model, frame, conf_threshold, nms_threshold, desired_classes)

        cv2.imshow("detections", frame)
        if save_output_video:
            writer.write(frame)