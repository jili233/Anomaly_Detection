# Anomaly Detection
## Dataset Preparation
Download the `images` folder from Gitlab, put it under the root folder.  
The structure of `images` folder should look like below:
```
images  #https://git.rwth-aachen.de/justin.pratt/ki-demonstrator/-/tree/main/images
├── blue
├── Fehler
├── Gutteile
├── red
└── yellow
```
## Training
Run `train.py` with the path of the dataset.   
For example,
```
python train.py /home/desktop/images/Gutteile
```
## Model Saving
After training, the weights of the model will be saved as `weights` in the root folder.

## Testing
Run `test.py` with the path of the image or the folder of the images to be tested. The reconstruction erorr will be printed in the terminal or saved under the folder.
For example,
```
python test.py /home/desktop/images/Gutteile
```
or
```
python test.py /home/desktop/images/Gutteile/opencv_frame_0.png
```