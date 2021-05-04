# Object Detection

This is a deep learning project, exploring object detection using [YOLOv5](https://github.com/ultralytics/yolov5) loaded via PyTorch Hub.


### Requirements
YOLOv5 requirements can be installed by:
```
$ pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt
```


### Model
The pre-trained model [YOLOv5s](https://github.com/ultralytics/yolov5#pretrained-checkpoints) is used.


### Data Set
*Preliminary dataset:* A subset of [Open Images Dataset V6](https://storage.googleapis.com/openimages/web/download.html) containing instrumental classes. The number of images and instances of each class is as follows:

| Class         | Images (Training) | Instances (Training)| Images (Validation) | Instances (Validation)| Images (Testing) | Instances (Testing)|
| ------------- | ----------------- | ------------------- | ------------------- | --------------------- | ---------------- | ------------------ |
| Accordion     | 839               | 955                 | 24                  | 24                    | 77               | 82
| Cello         | 1346              | 2004                | 27                  | 38                    | 78               | 86
| Piano         | 1246              | 1374                | 95                  | 100                   | 267              | 313
| Saxophone     | 854               | 1208                | 33                  | 40                    | 102              | 114
| Trumpet       | 835               | 1546                | 38                  | 65                    | 118              | 172
| Violin        | 1307              | 2028                | 29                  | 36                    | 93               | 101
