
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
python detect.py cfg/model_structure.cfg weights/000001.weights dataset/somfolder ../../dataset/voc2012/classes.name
```

Loading weights from yolov3.weights... Done!

data\dog.jpg: Predicted in 0.837523 seconds.  
3 box(es) is(are) found  
dog: 0.999996  
truck: 0.995232  
bicycle: 0.999973  
save plot results to predictions.jpg  

### validation and get evaluation results

```
python valid.py ../../Training.config ./cfg/model_structure.cfg ./weights/000001.weights
```



### License

MIT License (see LICENSE file).

