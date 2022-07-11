# USAGE
# python social_distance_detector.py --input pedestrians.mp4
# python social_distance_detector.py --input pedestrians.mp4 --output output.avi

from detectModule import config
from detectModule.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import os
import cv2


# Argument parsing for running on terminal
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
                help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
                help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
                help="output video should be displayed or not")
args = vars(ap.parse_args())


# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.join(config.MODEL_PATH, "coco.names")
LABELS = open(labelsPath).read().strip().split("\n")

weightsPath = os.path.join(config.MODEL_PATH, "yolov3.weights")
configPath = os.path.join(config.MODEL_PATH, "yolov3.cfg")

print("[INFO] loading YOLO from disk...")
yolo = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

if config.OPENCV_BACKEND:
    print("[NIFO] setting preferable backend to OpenCV and target to OPENCL...")
    yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)

layer_names = yolo.getLayerNames()
layer_names = [layer_names[i - 1] for i in yolo.getUnconnectedOutLayers()]

# initialize the video stream and pointer to the output video file
print("[INFO] accessing video stream...")
video = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None


# loop over the frames from the video stream
while True:
    
    ret, frame = video.read()

    if not ret:
        break

    frame = imutils.resize(frame, width=700)

    results = detect_people(frame, yolo, layer_names, personIdx=LABELS.index("person"))

    violate = set()

    if len(results) >= 2:
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric='euclidean')
        for i in range(0, D.shape[0]):
            for j in range(i+1, D.shape[1]):
                if D[i, j] < config.MIN_DISTANCE:
                    violate.add(i)
                    violate.add(j)

    for (i, (prob, bbox, centroid)) in enumerate(results):
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)

        if i in violate:
            color = (0, 0, 255)

        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame, (cX, cY), 5, color, 1)

    text = "Social Distancing Violations: {}".format(len(violate))
    cv2.putText(frame, text, (10, frame.shape[0] - 25),
              cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
  
    if args["display"] > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break

    if args["output"] != "" and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(args["output"], fourcc, 25, 
                                (frame.shape[1], frame.shape[0]), True)

    if writer is not None:
        writer.write(frame)