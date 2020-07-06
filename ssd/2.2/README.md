# This repository is created for implmentation of SSD (Single Shot Multibox Detector)

This code is forked from sgrvinod's code on github, and it has been modified to fit personal needs.
To make it more user-friendly, the hyperparameters of code are set in Training.config under Object_Detection rather than command string.

### After configure Training.config, Train your own data as follows:
run script create_data_lists.py first
```
1. python create_data_lists.py
2. python -W ignore train.py
```

* Saving  weights every epoch by default. 

* Basically, setwdata=False is enough for most users. 

* maximum epochs option, which is automatically calculated, somestimes is too small, then you can set the max_epochs in your configuration.



### Detect the objects in image using trained weights

#### SSD model
```
python -W ignore detect.py
```

### Validation and get evaluation results
```
python -W ignore eval.py
```



### License

MIT License (see LICENSE file).
