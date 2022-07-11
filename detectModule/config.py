# base path to YOLO directory
MODEL_PATH = '/home/thura/Desktop/TSF-internship/social-distancing-detector/yolo-coco/'

# initialize minimum probability to filter weak detections along-with
# the threshold when applying non-maxima supression
MIN_CONF = 0.3
NMS_THRESH = 0.3

# boolean indicating if OPENCV Backened & OPENCL Target should be used
OPENCV_BACKEND = True

# define the minimum safe distance (in pixels) that two people can be
# frome each other
MIN_DISTANCE = 50