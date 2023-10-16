import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from train import AutoEncoder
import argparse
import matplotlib.pyplot as plt
import csv

# Load pre-trained model
model = AutoEncoder()
model.load_state_dict(torch.load("weights.pth"))
model.eval()

# If you have a GPU available, move the model to GPU for faster computation
if torch.cuda.is_available():
    model = model.cuda()

parser = argparse.ArgumentParser(description='Process images from given path')
parser.add_argument('--data_path', type=str, required=True, help='Path to the image folder')
parser.add_argument("--classes", type=str, default='Fehler', help='CLasses used to test the model')
args = parser.parse_args()

# Data loader
transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

full_dataset = ImageFolder(root=args.data_path, transform=transform)

# Get the index of the class
classes_idx = full_dataset.class_to_idx[args.classes]

# Filter the dataset to only have images from the class
dataset = torch.utils.data.Subset(full_dataset, 
                [i for i, (_, label) in enumerate(full_dataset.samples) if label == classes_idx])

dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

def ae_loss(y_true, y_pred):
    return torch.mean((y_true - y_pred)**2)

losses = []

for idx, (imgs, _) in enumerate(dataloader):
    if torch.cuda.is_available():
        imgs = imgs.cuda()
    
    with torch.no_grad():
        reconstructions = model(imgs)

    for i, img in enumerate(imgs):
        loss = ae_loss(img.unsqueeze(0), reconstructions[i].unsqueeze(0)).item()

        img_idx = idx * dataloader.batch_size + i
        img_name = full_dataset.samples[img_idx][0].split("/")[-1]
        losses.append((img_name, loss))

# Save losses to a CSV file
csv_filename = f"reconstruction_losses_{args.classes}.csv"
with open(csv_filename, "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["ImageName", "Loss"])
    writer.writerows(losses)

print(f"Losses saved to {csv_filename}")

# Create a box plot for losses
loss_values = [loss for _, loss in losses]
plt.boxplot(loss_values)
plt.title(f"Boxplot for {args.classes} losses")
plt.ylabel('Loss Value')
plt.savefig(f"boxplot_{args.classes}.png")
plt.show()

print(f"Boxplot saved to boxplot_{args.classes}.png")