# Object_Detection
## Collection of implementation of object detection algorithm
This repository contains some useful object detection code, including YOLO v3, SSD.
It will integrate other object detection codes in the future. 


## Description:
### YOLO v3
* This code is forked from andy-yun's pytorch-0.4-yolov3 on github, and it has been modified to fit personal needs. 
* In training mode, avoid setting batch size = 1 because it will lead NAN value during gradient descent.
* For more detail, please refer to README.md of yolov3/5.1.0 

### SSD
* This code is forked from sgrvinod on github, and it has been modified to fit personal needs. 
* Before training stage, run create_data_lists.py to create json files.
* For more detail, please refer to README.md of ssd/5.1.0 
