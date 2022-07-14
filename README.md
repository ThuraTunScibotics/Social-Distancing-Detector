# Social-Distancing-Detector
## Implementation of COVID-19 social distancing detector using OpenCV, Computer Vision and Deep Learning.

## Background - One of the tasks for Computer Vision internship at [The Sparks Foundation](https://internship.thesparksfoundation.info/).

## Algorithm
1. Detection of people from input frame using YOLOv3
2. Compute the Euclidean distance between all pairs of the centroid (pairwise distances)
3. Check the distance matrix < configured distance
    * if distance < configured_distance, add distance_coordinate to violation set
    * set different color for bounding-boxes of each situation for violation or not-violation
    * Put number of violations status  on the resultant frame
4. Show and Write frames for output video

## Requirements
Check package manager, [anaconda]() which will be required to install required libraries & packages under specific virtual environment.
Install anaconda on your machine, and run the following cell on terminal/command prompt after installed.
```
conda create -n SocialDistancingDetector jupyter python opencv imutils scipy numpy pandas matplotlib
```

## Running & Demo
Open Terminal run the following under actvated environment with preinstalled required libs and packages.
```
python social_distance_detector.py --input pedestrians.mp4 --output output.avi
```
Demo

![Demo](https://github.com/ThuraTunScibotics/Social-Distancing-Detector/blob/main/output.gif)

## Reference

https://analyticsindiamag.com/covid-19-computer-vision/

https://pyimagesearch.com/2020/06/01/opencv-social-distancing-detector/
