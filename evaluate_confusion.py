import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input_tensor):
        batch_size, channels, _, _ = input_tensor.size()
        y = self.avg_pool(input_tensor).view(batch_size, channels)
        y = self.fc(y).view(batch_size, channels, 1, 1)
        return input_tensor * y.expand_as(input_tensor)


class CustomResNet50SE(nn.Module):
    def __init__(self, num_classes=100, dropout_rate=0.5):
        super().__init__()
        backbone = models.resnet50(weights=None)
        
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        self.se1 = SqueezeExcitation(256)
        self.se2 = SqueezeExcitation(512)
        self.se3 = SqueezeExcitation(1024)
        self.se4 = SqueezeExcitation(2048)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(2048, num_classes)
        )

    def forward(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.se1(x)
        
        x = self.layer2(x)
        x = self.se2(x)
        
        x = self.layer3(x)
        x = self.se3(x)
        
        x = self.layer4(x)
        x = self.se4(x)

        x = self.global_pool(x)
        logits = self.classifier_head(x)
        
        return logits


def get_validation_loader(valid_dir, batch_size=32):
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    
    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std)
    ])
    
    valid_dataset = datasets.ImageFolder(root=valid_dir, transform=valid_transform)
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=torch.cuda.is_available()
    )
    
    return valid_loader, valid_dataset.classes


def generate_predictions(model, data_loader, device):
    model.eval()
    all_predictions = []
    all_true_labels = []
    
    progress_bar = tqdm(data_loader, desc="Evaluating Validation Set")
    
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(labels.numpy())
            
    return np.array(all_true_labels), np.array(all_predictions)


def extract_top_confusions(conf_matrix, class_names, top_n=10):
    """Finds the most frequently confused pairs, ignoring correct predictions."""
    # Create a copy and zero out the diagonal (correct predictions)
    matrix_no_diag = conf_matrix.copy()
    np.fill_diagonal(matrix_no_diag, 0)
    
    # Get the indices of the highest values in the flattened matrix
    flat_indices = np.argsort(matrix_no_diag, axis=None)[::-1]
    
    print(f"\n--- Top {top_n} Most Confused Class Pairs ---")
    
    printed_count = 0
    for idx in flat_indices:
        if printed_count >= top_n:
            break
            
        # Convert flat index back to 2D row/col indices
        true_idx, pred_idx = np.unravel_index(idx, matrix_no_diag.shape)
        confusion_count = matrix_no_diag[true_idx, pred_idx]
        
        if confusion_count == 0:
            break # No more errors to report
            
        true_class = class_names[true_idx]
        pred_class = class_names[pred_idx]
        
        print(f"True: '{true_class}' | Predicted: '{pred_class}' | Count: {confusion_count}")
        printed_count += 1


def plot_and_save_matrix(conf_matrix, class_names, output_filename="confusion_matrix.png"):
    """Saves a high-resolution heatmap of the confusion matrix."""
    plt.figure(figsize=(24, 20)) # Very large figure to accommodate 100 classes
    
    # Use a logarithmic color scale to make minor errors visible alongside massive correct predictions
    sns.heatmap(
        conf_matrix, 
        annot=False, # Set to True if you want numbers in the boxes, but it gets messy with 100 classes
        cmap="Blues", 
        xticklabels=class_names, 
        yticklabels=class_names,
        linewidths=0.1,
        linecolor='gray'
    )
    
    plt.title('Validation Confusion Matrix', fontsize=24)
    plt.ylabel('True Class', fontsize=18)
    plt.xlabel('Predicted Class', fontsize=18)
    
    # Rotate tick labels for readability
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"\nHigh-resolution confusion matrix saved to '{output_filename}'")


if __name__ == "__main__":
    valid_directory = "./data/valid"
    model_checkpoint = "best_swa_resnet50_model.pth"
    num_categories = 100
    
    compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {compute_device}")
    
    if not os.path.exists(valid_directory):
        print(f"Error: Directory '{valid_directory}' not found.")
        exit(1)
        
    loader, categories = get_validation_loader(valid_directory)
    
    classification_model = CustomResNet50SE(num_classes=num_categories)
    classification_model.load_state_dict(torch.load(model_checkpoint, map_location=compute_device))
    classification_model.to(compute_device)
    
    true_targets, predictions = generate_predictions(classification_model, loader, compute_device)
    
    # Calculate the mathematical matrix
    c_matrix = confusion_matrix(true_targets, predictions)
    
    # Output the actionable text analysis
    extract_top_confusions(c_matrix, categories, top_n=15)
    
    # Output the visual heatmap
    plot_and_save_matrix(c_matrix, categories)