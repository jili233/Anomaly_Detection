import torch
import os
import json
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from train import AutoEncoder

# Load pre-trained model
model = AutoEncoder()
model.load_state_dict(torch.load("weights.pth"))
model.eval()

# If you have a GPU available, move the model to GPU for faster computation
if torch.cuda.is_available():
    model = model.cuda()

# Data loader
transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = ImageFolder(root="/home/jili/Desktop/images/Fehler", transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

def ae_loss(y_true, y_pred):
    return torch.mean((y_true - y_pred)**2)

losses = {}

# Iterate over each image, reconstruct, and compute loss
for idx, (img, _) in enumerate(dataloader):
    if torch.cuda.is_available():
        img = img.cuda()
    
    with torch.no_grad():
        reconstruction = model(img)
        loss = ae_loss(img, reconstruction).item()
        
    # Assuming your images have filenames you want to use
    img_name = dataset.samples[idx][0].split("/")[-1]  # Extracting filename
    losses[img_name] = loss

# Save losses to a JSON file
with open("reconstruction_losses.json", "w") as file:
    json.dump(losses, file)

print("Losses saved to reconstruction_losses.json")