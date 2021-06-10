import yolov4
import numpy as np

# Variables declaration
TEST_VIDEO_FILE_NAME = "video/Zona 16.mp4"
MIN_CONFIDENCE_THRESHOLD = 0.5
SAVE_OUTPUT_VIDEO = True
OUTPUT_VIDEO_FPS = 30

SKIP_FRAMES_TO_SPEEDUP = True
SKIP_FRAME_COUNT = 20
DETECT_ALL_CLASSES = False
desiredClasses = ["person"]

SAVE_DETECTED_OBJECT_IMAGES = False

# Creating net
model = yolov4.setup_network(416,416,1/255)
# Performing detection on video
yolov4.detect_in_video_with_yolov4(model,TEST_VIDEO_FILE_NAME, 30, MIN_CONFIDENCE_THRESHOLD, 0.4, desiredClasses, True, "zona_16_trial")

# Just a comment to try git bash in push