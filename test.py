import torch
import csv
import argparse
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import torchvision.models as models
from torch import nn

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Backbone class (same as in the training script)
class Backbone(nn.Module):
    def __init__(self, backbone_name='resnet18', num_classes=2, class_weights=[5.0, 1.0]):
        super(Backbone, self).__init__()

        # Choose the backbone
        if hasattr(models, backbone_name):
            self.features = getattr(models, backbone_name)(pretrained=True)
        else:
            raise ValueError(f"Model {backbone_name} not found in torchvision.models")

        for param in self.features.parameters():
            param.requires_grad = False
        
        in_features = self.features.fc.in_features
        self.features.fc = nn.Linear(in_features, num_classes)

        # adopted loss function CrossEntropyLoss
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
    
    def forward(self, x):
        return self.features(x)

def load_weights(model, path):
    model.load_state_dict(torch.load(path))

def calculate_precision_recall_f1(y_true, y_pred, class_idx):
    tp = torch.sum((y_true == class_idx) & (y_pred == class_idx)).item()
    fp = torch.sum((y_true != class_idx) & (y_pred == class_idx)).item()
    fn = torch.sum((y_true == class_idx) & (y_pred != class_idx)).item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

def infer(model, loader, full_dataset, subset_indices):
    model.eval()
    results = []
    all_labels = []
    all_preds = []
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            logits = model(images)
            probabilities = softmax(logits)
            _, preds = torch.max(logits, 1)

            for i, (probability, pred, label) in enumerate(zip(probabilities, preds, labels)):
                subset_index = subset_indices[batch_idx * loader.batch_size + i]
                img_path = full_dataset.imgs[subset_index][0]
                results.append({
                    'image_path': os.path.basename(img_path),
                    'predicted_probability': probability[pred.item()].item(),
                    'predicted_class': full_dataset.classes[pred.item()],
                    'actual_class': full_dataset.classes[label.item()],
                    'correct': pred.item() == label.item()
                })
                all_labels.append(label.item())
                all_preds.append(pred.item())

    return results, all_labels, all_preds

def write_results_to_csv(results, file_path):
    with open(file_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

def main():
    parser = argparse.ArgumentParser(description='PyTorch Inference with Command Line Argument for Data Path')
    parser.add_argument("--data_path", type=str, required=True, help='Path to the data')
    parser.add_argument("--weights_path", type=str, default=os.path.join(current_dir, 'weights.pth'), help='Path to the model weights')
    parser.add_argument("--output_path", type=str, default=os.path.join(current_dir, 'output.csv'), help='Path to save the CSV output')
    parser.add_argument("--batch_size", type=int, default=8, help='Batch size for testing')
    parser.add_argument("--img_size", type=int, nargs=2, default=[640, 480], help='Image size as tuple (width, height)')
    parser.add_argument("--backbone", type=str, default='resnet18', help='Backbone model name')
    parser.add_argument("--class_weights", nargs=2, type=float, default=[5.0, 1.0], help='Weights for classes')
    args = parser.parse_args()

    # Data preparation
    transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
    ])
    full_dataset = ImageFolder(root=args.data_path, transform=transform)
    class_names = full_dataset.classes
    # Get the indices of the classes "Gutteile" and "Fehler"
    gutteile_idx = full_dataset.class_to_idx['Gutteile']
    fehler_idx = full_dataset.class_to_idx['Fehler']
    
    # Filter the dataset to only have images from the classes "Gutteile" and "Fehler"
    filtered_indices = [i for i, (_, label) in enumerate(full_dataset.imgs) if label in [gutteile_idx, fehler_idx]]
    filtered_dataset = torch.utils.data.Subset(full_dataset, filtered_indices)

    loader = DataLoader(filtered_dataset, batch_size=args.batch_size, shuffle=False)

    # Model setup
    model = Backbone(backbone_name=args.backbone, num_classes=2, class_weights=args.class_weights)
    load_weights(model, args.weights_path)

    # Inference
    results, all_labels, all_preds = infer(model, loader, full_dataset, filtered_indices)


    # Calculate metrics for each class
    classes = full_dataset.classes
    metrics = {}
    for i, class_name in enumerate(classes):
        precision, recall, f1 = calculate_precision_recall_f1(torch.tensor(all_labels), torch.tensor(all_preds), i)
        metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    # Print the metrics for each class
    for class_name, scores in metrics.items():
        print(f"Class: {class_name}")
        print(f"Precision: {scores['precision']}")
        print(f"Recall: {scores['recall']}")
        print(f"F1 Score: {scores['f1']}\n")

    # Save results to CSV
    write_results_to_csv(results, args.output_path)

if __name__ == '__main__':
    main()