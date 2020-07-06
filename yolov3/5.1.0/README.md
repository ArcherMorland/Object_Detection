
# This repository is created for implmentation of yolov3 with pytorch 0.4
This code is forked from andy-yun's pytorch-0.4-yolov3 on github, and it has been modified to fit personal needs.
To make it more user-friendly, the hyperparameters of code are set in Training.config under Object_Detection rather than command string.


### After configure Training.config, Train your own data as follows:
```
python -W ignore train.py
```

* last 10 weights are saved in weights directory 

* If you want to  train weights from pretrained yolov3.weights, go to cfg/setting.config and set training_process_init=True and check dataset is properly placed then you're good to go .
   Use the following link to download pretrained yolov3.weight:
   ```
   https://pjreddie.com/media/files/yolov3.weights
   ```

* Basically, setwdata=False is enough for most users. 

* maximum epochs option, which is automatically calculated, somestimes is too small, then you can set the max_epochs in your configuration.



### Detect the objects in dog image using pretrained weights


#### yolov3 models
```
wget https://pjreddie.com/media/files/yolov3.weights
python detect.py cfg/yolo_v3.cfg yolov3.weights data/dog.jpg data/coco.names  
```

![predictions](data/predictions-yolov3.jpg)

Loading weights from yolov3.weights... Done!

data\dog.jpg: Predicted in 0.837523 seconds.  
3 box(es) is(are) found  
dog: 0.999996  
truck: 0.995232  
bicycle: 0.999973  
save plot results to predictions.jpg  

### validation and get evaluation results

```
valid.py data/yourown.data cfg/yourown.cfg yourown_weights
```

### Performances for voc datasets using yolov2 (with 100 epochs training)
- CrossEntropyLoss is used to compare classes
- Performances are varied along to the weighting factor, for example.
```
coord_scale=1, object_scale=5, class_scale=1 mAP = 73.1  
coord_scale=1, object_scale=5, class_scale=2 mAP = 72.7  
coord_scale=1, object_scale=3, class_scale=1 mAP = 73.4  
coord_scale=1, object_scale=3, class_scale=2 mAP = 72.8  
coord_scale=1, object_scale=1, class_scale=1 mAP = 50.4  
```

- After modifying anchors information at yolo-voc.cfg and applying new coord_mask
Finally, I got the 
```
anchors = 1.1468, 1.5021, 2.7780, 3.4751, 4.3845, 7.0162, 8.2523, 4.2100, 9.7340, 8.682
coord_scale=1, object_scale=3, class_scale=1 mAP = 74.4  
```

- using yolov3 with self.rescore = 1 and latest code, ___mAP = 74.9___. (with 170 epochs training)

Therefore, you may do many experiments to get the best performances.

### License

MIT License (see LICENSE file).
