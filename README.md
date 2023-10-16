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
Run `train.py` with the path of the `images` folder and the classes of the images for training(default is Gutteile).
```
python train.py /home/desktop/images --classes Gutteile
```
## Model Saving
After training, the weights of the model will be saved as `weights.pth` under the root folder.

## Testing
Run `test.py` with the path of the `images` folder and the classes of the images for testing(default is Fehler). The reconstruction erorr will be saved as a `.csv` file under the root folder.
```
python test.py --data_path /home/usr/images --classes Fehler
```