# Anomaly Detection

## Installation
`pip install lightning torch torchvision`
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
## Current Training Results
<img src="Screenshot from 2023-10-28 15-06-55.png" width="" alt=""/>

## Training
Run `train.py` with the path of the `images` folder.
```
python train.py --data_path /home/desktop/images
```
## Model Saving
After training, the weights of the model will be saved as `weights.pth` under the root folder.

## Testing
Run `test.py` with the path of the `images` folder. The test will be conducted on all the images (Gutteile+Fehler) and results will be saved as a `.csv` file under the root folder. `weights.pth` needs to be placed under the root folder. 
```
python test.py --data_path /home/usr/images
```
