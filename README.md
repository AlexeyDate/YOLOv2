# Implementation YOLOv2 using PyTorch 
![image1](https://user-images.githubusercontent.com/86290623/228918454-ae2f39dd-4ccc-4f55-93a1-fcea701f4132.jpg)


## Dataset
* This repository was train on the [African Wildlife Dataset](https://www.kaggle.com/datasets/biancaferreira/african-wildlife) from Kaggle
* The data folder must contain the train and test folders as follows:
> 
    ├── data 
      ├── train
        ├── class1
          ├── 001.jpg
          ├── 001.txt(xml)
      ├── test 
        ├── class1
          ├── 001.jpg
          ├── 001.txt(xml)
      ├── obj.data
      ├── obj.names

* Also, for training, in the data folder there must be `obj.data` file with some settings
>
    classes = 4
    train = data/train
    valid = data/test
    names = data/obj.names
    backup = backup/
    file_format = txt
    convert_to_yolo = False
    
* And there must be `obj.names` file with label names.
>
    0 buffalo
    1 elephant
    2 rhino
    3 zebra

* In the description files, you can write the coordinates of the bounding boxes in a simple format `(x1, y1, x2, y2)`. After this use the appropriate flag when training. YOLO format is also available and recommended. Format files as follows:    
    
**txt**
>
    <class> <xmin> <ymin> <xmax> <ymax>
example:
>
    1 207 214 367 487
___
**txt** (already converted to yolo)
>
    <class> <xcenter> <ycenter> <width> <height>
example:
>
    1 0.2 0.3 0.15 0.23
___
**xml**

example:
>
    <annotation>
	<object>
		<name>zebra</name>
		<bndbox>
			<xmin>71</xmin>
			<ymin>60</ymin>
			<xmax>175</xmax>
			<ymax>164</ymax>
		</bndbox>
    
## Training
> 
    python3 train.py --epochs 100 --learning_rate 1e-5 
    
All training parameters:

`--epochs`                  (states: total epochs)

`--learning_rate`           (states: learning rate)

`--batch_size`              (states: batch size)

`--weight_decay`            (states: weight decay)

`--yolo_weights`            (states: path to yolo PyTorch weights)

`--darknet_weights`         (states: path to extraction binary weights, it's base CNN module of YOLO)

`--multiscale_off`          (states: disable multi-scale training)

After training, mAP will be calculated on the train dataloader and the test dataloader.
You can change the thresholds in `train.py`.

## Inference
On video:
> 
    python3 detect.py --video --data_test content/video.mp4 --output content/detect.mp4 --weights backup/yolov1.pt
On image:
> 
    python3 detect.py --data_test content/image.jpg --output content/detect.jpg --weights backup/yolov1.pt

Additional parameters:

`--show`          (states: show frames during inference)

![image2](https://user-images.githubusercontent.com/86290623/228927611-e747d106-19ba-4bcd-8d8a-5435b99bb89b.jpg)

## Comparison
| Model   		      | Dataset 	   |Input size <br> <sub> (pixel)   | mAP <br> <sub>(@0.5)   |
| :---:   		      | :---:   	   | :---:    	                    | :---: 		     |
| YOLOv1 <br> <sub> (Ours⭐)  | African Wildlife   | 416       	                   | 70     	  	    |
| YOLOv2 <br> <sub> (Ours⭐)  | African Wildlife   | 448       	      	           | 61    	            |

## Dependencies
**PyTorch** 
> Version: 1.13.1

**Albumentations**
> Version: 1.3.0

**OpenCV**
> Version: 4.7.0

**xmltodict**
> Version: 0.13.0
		
## Specificity
* Training may not be established in the first epochs.
* You should be careful with multi-scale training.

## References
* [Original YOLOv2 paper](https://arxiv.org/pdf/1506.02640.pdf)
___
* [Darknet19 weights from ImageNet](https://pjreddie.com/media/files/darknet19_448.weights) (recommended for all trainings)
___
* [African Wildlife dataset](https://www.kaggle.com/datasets/biancaferreira/african-wildlife?resource=download)
* [African Wildlife PyTorch weights](https://drive.google.com/file/d/1-0xX8dxh4oc6FhGH3jighnMQDLNMrR67/view?usp=share_link)
* [African Wildlife optimizer state](https://drive.google.com/file/d/1-3mbEJSViHkMYB9Ru4x05h0kx9Tou7cn/view?usp=share_link)
