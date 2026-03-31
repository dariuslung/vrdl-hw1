import os
import csv
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image


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


class FlatImageDataset(Dataset):
    def __init__(self, directory_path, transform=None):
        self.directory_path = directory_path
        self.transform = transform
        self.valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".tif", ".tiff"}
        
        self.image_filenames = sorted([
            file_name for file_name in os.listdir(directory_path)
            if os.path.splitext(file_name)[1].lower() in self.valid_extensions
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        file_name = self.image_filenames[index]
        file_path = os.path.join(self.directory_path, file_name)
        
        image_data = Image.open(file_path).convert("RGB")
        
        if self.transform is not None:
            image_data = self.transform(image_data)
            
        image_name_no_ext = os.path.splitext(file_name)[0]
        
        return image_data, image_name_no_ext


class StackAndNormalizeCrops:
    def __init__(self, mean, std):
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, crops):
        return torch.stack([self.normalize(self.to_tensor(crop)) for crop in crops])


def load_inference_model(num_classes, checkpoint_path, device):
    model = CustomResNet50SE(num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def run_ensemble_inference(test_directory, model_a_path, model_b_path, output_csv_path, num_classes=100, batch_size=16):
    """
    Runs Soft Voting ensemble inference across two models.
    Batch size is reduced to 16 to accommodate loading two models into VRAM simultaneously.
    """
    compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running ensemble inference on device: {compute_device}")
    
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    
    inference_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.TenCrop(224),
        StackAndNormalizeCrops(mean=image_mean, std=image_std)
    ])
    
    test_dataset = FlatImageDataset(directory_path=test_directory, transform=inference_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print("Loading Model A (Baseline)...")
    model_a = load_inference_model(num_classes, model_a_path, compute_device)
    
    print("Loading Model B (SWA)...")
    model_b = load_inference_model(num_classes, model_b_path, compute_device)
    
    predictions_list = []
    class_mapping = torch.load("class_mapping.pth", map_location=compute_device)
    
    print(f"Starting ensemble inference on {len(test_dataset)} images using 10-Crop TTA...")
    
    with torch.no_grad():
        for images, image_names in test_loader:
            bs, n_crops, c, h, w = images.size()
            images = images.view(-1, c, h, w).to(compute_device)
            
            # Get logits from Model A
            outputs_a = model_a(images)
            outputs_a = outputs_a.view(bs, n_crops, -1).mean(dim=1)
            
            # Get logits from Model B
            outputs_b = model_b(images)
            outputs_b = outputs_b.view(bs, n_crops, -1).mean(dim=1)
            
            # Soft Voting: Average the logits
            # Note: You can apply weighting here if desired (e.g., outputs_a * 0.6 + outputs_b * 0.4)
            ensemble_outputs = (outputs_a + outputs_b) / 2.0
            
            _, predicted_classes = torch.max(ensemble_outputs, 1)
            
            for img_name, pred_class in zip(image_names, predicted_classes):
                actual_class_name = class_mapping[pred_class.item()]
                predictions_list.append((img_name, actual_class_name))
                
    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["image_name", "pred_label"])
        for img_name, pred_class in predictions_list:
            csv_writer.writerow([img_name, pred_class])
            
    print(f"Ensemble inference complete. Predictions saved to '{output_csv_path}'.")


if __name__ == "__main__":
    target_test_directory = "./data/test" 
    
    # Define paths to both your top-performing models
    baseline_checkpoint = "best_custom_resnet50_model.pth"
    swa_checkpoint = "best_swa_resnet50_model.pth"
    output_filename = "ensemble_prediction.csv"
    target_categories = 100
    
    if not os.path.exists(target_test_directory):
        print(f"Error: Target directory '{target_test_directory}' not found.")
    else:
        run_ensemble_inference(
            test_directory=target_test_directory,
            model_a_path=baseline_checkpoint,
            model_b_path=swa_checkpoint,
            output_csv_path=output_filename,
            num_classes=target_categories,
            batch_size=16 # Kept slightly lower to prevent out-of-memory errors with two models loaded
        )