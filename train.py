import pytorch_lightning as pl
import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import argparse
import torchvision.models as models
from pytorch_lightning.callbacks import ProgressBar

class CustomProgressBar(ProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.set_description('Epoch')
        return bar

class Backbone(pl.LightningModule):
    def __init__(self, num_classes=2):
        super(Backbone, self).__init__()

        self.features = models.resnet18(pretrained=True)
        
        for param in self.features.parameters():
            param.requires_grad = False
        
        in_features = self.features.fc.in_features
        self.features.fc = nn.Linear(in_features, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.features(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        self.log('test_acc', acc, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def accuracy(self, logits, labels):
        _, preds = torch.max(logits, 1)
        return torch.sum(preds == labels.data).float() / len(labels)

class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=8, img_size=(640, 480), test_split=0.2, num_workers=15):
        super(DataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.test_split = test_split
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
        ])

    def setup(self, stage=None):
        # Load the dataset using ImageFolder
        full_dataset = ImageFolder(root=self.data_dir, transform=self.transform)
        
        # Get the indices of the classes "Gutteile" and "Fehler"
        gutteile_idx = full_dataset.class_to_idx['Gutteile']
        fehler_idx = full_dataset.class_to_idx['Fehler']
        
        # Filter the dataset to only have images from the classes "Gutteile" and "Fehler"
        filtered_indices = [i for i, (_, label) in enumerate(full_dataset.samples) if label in [gutteile_idx, fehler_idx]]
        self.dataset = torch.utils.data.Subset(full_dataset, filtered_indices)
        
        # Split the dataset into train and test sets
        dataset_size = len(self.dataset)
        test_size = int(self.test_split * dataset_size)
        train_size = dataset_size - test_size
        self.train_dataset, self.test_dataset = random_split(self.dataset, [train_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

def main():
    parser = argparse.ArgumentParser(description='PyTorch Encoder-Decoder with Command Line Argument for Data Path')
    parser.add_argument("--data_path", type=str, required=True, help='Path to the data')
    parser.add_argument("--accelerator", default='cpu')
    args = parser.parse_args()
    
    data_module = DataModule(data_dir=args.data_path)
    model = Backbone(num_classes=2)
    
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator=args.accelerator,
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
    torch.save(model.state_dict(), 'weights.pth')

if __name__ == '__main__':
    main()