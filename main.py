# import yolov4
import format_dataset
import metrics

# # Variables declaration
# TEST_VIDEO_FILE_NAME = "video/Zona 16.mp4"
# MIN_CONFIDENCE_THRESHOLD = 0.5
# SAVE_OUTPUT_VIDEO = True
# OUTPUT_VIDEO_FPS = 30

# SKIP_FRAMES_TO_SPEEDUP = True
# SKIP_FRAME_COUNT = 5
# DETECT_ALL_CLASSES = False
# desiredClasses = ["person"]

# SAVE_DETECTED_OBJECT_IMAGES = False

# PROCESSING 
#image_path = r'C:\Users\Riccardo\Desktop\TESI MAGISTRALE\Code\VISDRONE\original_visdrone_only_pedestrian_and_people\VisDrone2019-DET-test-dev\images'
annotations_path = r'C:\Users\Riccardo\Desktop\TESI MAGISTRALE\Code\VISDRONE\original_visdrone_only_pedestrian_and_people\VisDrone2019-DET-test-dev\annotations'
annotations_formatted_path = r'C:\Users\Riccardo\Desktop\TESI MAGISTRALE\Code\VISDRONE\original_visdrone_only_pedestrian_and_people\VisDrone2019-DET-test-dev\annotations\_annotations.txt'
#format_dataset.format_annotations_visdrone_to_yolov4_pytorch(annotations_path,0)
#format_dataset.remove_not_corresponding(image_path,annotations_formatted_path)
# DRAW BOUNDING BOXES 
image_path = r'C:\Users\Riccardo\Desktop\TESI MAGISTRALE\Code\master_degree\visualize_bbox\0000006_01111_d_0000003.jpg'
annotations = r'C:\Users\Riccardo\Desktop\TESI MAGISTRALE\Code\master_degree\visualize_bbox\annots.txt'
#format_dataset.format_annotations_SARD_to_yolov4_pytorch(r'C:\Users\Riccardo\Desktop\TESI MAGISTRALE\Code\SARD\SARD',r'C:\Users\Riccardo\Desktop\TESI MAGISTRALE\Code\SARD\SARD\formatted_anno.txt')
format_dataset.visualize_bounding_box(image_path,annotations)


# TESTING METRICS 
#ground_truth_annotations = r'C:\Users\Riccardo\Desktop\TESI MAGISTRALE\Code\master_degree\metrics\_annotations.txt'
#metrics.IoU_dataset(ground_truth_annotations)