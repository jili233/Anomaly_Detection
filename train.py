import pytorch_lightning as pl
import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import argparse

class AEDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=128, img_size=(256, 256)):
        super(AEDataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    def setup(self, stage=None):
        # Load the dataset using ImageFolder
        full_dataset = ImageFolder(root=self.data_dir, transform=self.transform)
        
        # Get the index of the 'Gutteile' class
        gutteile_idx = full_dataset.class_to_idx['Gutteile']
        
        # Filter the dataset to only have images from the 'Gutteile' class
        self.dataset = torch.utils.data.Subset(full_dataset, 
                       [i for i, (_, label) in enumerate(full_dataset.samples) if label == gutteile_idx])

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )
        
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 8 * 8, 200)
        
    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.conv_block_5(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.fc = nn.Linear(200, 64 * 8 * 8)
        self.reshape = lambda x: x.view(-1, 64, 8, 8)
        
        self.deconv_block_1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        
        self.deconv_block_2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        
        self.deconv_block_3 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        
        self.deconv_block_4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )
        
        self.deconv_block_5 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        
    def forward(self, x):
        x = self.fc(x)
        x = self.reshape(x)
        x = self.deconv_block_1(x)
        x = self.deconv_block_2(x)
        x = self.deconv_block_3(x)
        x = self.deconv_block_4(x)
        x = self.deconv_block_5(x)
        return torch.sigmoid(x)


class AutoEncoder(pl.LightningModule):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)
    
    def ae_loss(self, y_true, y_pred):
        # Matching the original ae_loss calculation
        loss = torch.mean((y_true - y_pred) ** 2, dim=[1, 2, 3])
        return loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = self.ae_loss(x, x_hat)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0005)

def main():
    parser = argparse.ArgumentParser(description='PyTorch Encoder-Decoder with Command Line Argument for Data Path')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data')
    args = parser.parse_args()
    data_path = args.data_path
    
    # Training
    data_module = AEDataModule(data_path)
    model = AutoEncoder()
    trainer = pl.Trainer(max_epochs=50)
    trainer.fit(model, datamodule=data_module)
    torch.save(model.state_dict(), 'weights.pth')

if __name__ == '__main__':
    main()