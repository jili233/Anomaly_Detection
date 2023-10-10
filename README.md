# Anomaly Detection
## Dataset Preparation
Download the images folder from Gitlab, put it under the root folder of this project.
The structure of ‘images’ folder should look like below:
- [images]([https://pjreddie.com/projects/pascal-voc-dataset-mirror/](https://git.rwth-aachen.de/justin.pratt/ki-demonstrator/-/tree/main/images))
  - blue
  - Fehler
  - Gutteile
  - red
  - yellow

## Training
Run 'en-de.py' with the path of the dataset. For example,
'''
python en-de.py /home/desktop/images/Gutteile
'''

## Model Saving
After Training, the weights of the model will be saved as
