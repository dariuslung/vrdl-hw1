import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import WeightedRandomSampler, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


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
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        # ResNet-50 uses bottleneck blocks, altering the output channels
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


def create_balanced_sampler(train_dataset):
    num_classes = len(train_dataset.classes)
    class_counts = [0] * num_classes
    
    for _, label in train_dataset.samples:
        class_counts[label] += 1
        
    class_weights = [1.0 / count for count in class_counts]
    sample_weights = [class_weights[label] for _, label in train_dataset.samples]
    sample_weights_tensor = torch.DoubleTensor(sample_weights)
    
    balanced_sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True
    )
    
    return balanced_sampler


def get_data_loaders(dataset_base_path, batch_size=32, num_workers=4):
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    normalize_transform = transforms.Normalize(mean=image_mean, std=image_std)

    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandAugment(),
            transforms.ToTensor(),
            normalize_transform,
            transforms.RandomErasing(p=0.2)
        ]),
        "valid": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize_transform
        ])
    }

    train_dir = os.path.join(dataset_base_path, "train")
    valid_dir = os.path.join(dataset_base_path, "valid")

    image_datasets = {
        "train": datasets.ImageFolder(root=train_dir, transform=data_transforms["train"]),
        "valid": datasets.ImageFolder(root=valid_dir, transform=data_transforms["valid"]),
    }

    torch.save(image_datasets["train"].classes, "class_mapping.pth")
    
    use_pin_memory = torch.cuda.is_available()
    sampler = create_balanced_sampler(image_datasets["train"])

    data_loaders = {
        "train": DataLoader(
            image_datasets["train"],
            batch_size=batch_size,
            shuffle=False, 
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=use_pin_memory
        ),
        "valid": DataLoader(
            image_datasets["valid"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin_memory
        )
    }

    return data_loaders


def initialize_model_and_optimizer(num_classes, learning_rate):
    model = CustomResNet50SE(num_classes=num_classes)
    
    # Initial phase: Freeze all layers except layer4, SE blocks, and classification head
    for name, param in model.named_parameters():
        if "classifier_head" not in name and "se" not in name and "layer4" not in name:
            param.requires_grad = False
            
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    
    loss_criterion = nn.CrossEntropyLoss()
    model_optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-4)
    
    lr_scheduler = ReduceLROnPlateau(
        model_optimizer, 
        mode='min', 
        factor=0.1, 
        patience=3
    )
    
    return model, loss_criterion, model_optimizer, lr_scheduler


def update_optimizer_for_unfreezing(model, learning_rate):
    """Re-initializes the optimizer and scheduler when new layers are unfrozen."""
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    new_optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-4)
    new_scheduler = ReduceLROnPlateau(
        new_optimizer, 
        mode='min', 
        factor=0.1, 
        patience=3
    )
    return new_optimizer, new_scheduler


def train_one_epoch(model, data_loader, loss_criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(data_loader, desc="Training", leave=False)
    
    for inputs, labels in progress_bar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        current_batch_size = inputs.size(0)
        running_loss += loss.item() * current_batch_size
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        
        current_loss = running_loss / total_samples
        current_acc = correct_predictions / total_samples
        progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.4f}")
        
    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples
    
    return epoch_loss, epoch_accuracy


def evaluate_model(model, data_loader, loss_criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(data_loader, desc="Validating", leave=False)
    
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = loss_criterion(outputs, labels)
            
            current_batch_size = inputs.size(0)
            running_loss += loss.item() * current_batch_size
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            current_loss = running_loss / total_samples
            current_acc = correct_predictions / total_samples
            progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.4f}")
            
    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples
    
    return epoch_loss, epoch_accuracy


def plot_training_metrics(train_losses, valid_losses, train_accs, valid_accs):
    epochs_range = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.plot(epochs_range, valid_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accs, label='Training Accuracy')
    plt.plot(epochs_range, valid_accs, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()


if __name__ == "__main__":
    target_categories = 100
    dataset_directory = "./data"
    total_epochs = 35
    initial_learning_rate = 0.001
    
    # Progressive Unfreezing Configuration
    unfreeze_layer3_epoch = 12
    unfreeze_layer2_epoch = 22
    
    compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    history_train_loss = []
    history_valid_loss = []
    history_train_acc = []
    history_valid_acc = []
    
    if not os.path.exists(dataset_directory):
        exit(1)
        
    loaders = get_data_loaders(dataset_directory)
    classification_model, criterion, optimizer, scheduler = initialize_model_and_optimizer(
        num_classes=target_categories,
        learning_rate=initial_learning_rate
    )
    classification_model = classification_model.to(compute_device)
    
    best_valid_loss = float('inf')
    model_save_path = "best_custom_resnet50_model.pth"
    tensorboard_writer = SummaryWriter("runs/custom_resnet50_experiment_01")
    
    for epoch in range(1, total_epochs + 1):
        print(f"\nEpoch {epoch}/{total_epochs}")
        print("-" * 20)
        
        # Progressive Unfreezing Logic
        if epoch == unfreeze_layer3_epoch:
            print(f"\n[Epoch {epoch}] Unfreezing layer3...")
            for param in classification_model.layer3.parameters():
                param.requires_grad = True
            # Re-initialize optimizer with a lower learning rate for fine-tuning
            optimizer, scheduler = update_optimizer_for_unfreezing(classification_model, learning_rate=1e-4)
            
        elif epoch == unfreeze_layer2_epoch:
            print(f"\n[Epoch {epoch}] Unfreezing layer2 and layer1...")
            for param in classification_model.layer2.parameters():
                param.requires_grad = True
            for param in classification_model.layer1.parameters():
                param.requires_grad = True
            # Further reduce learning rate as deeper layers are updated
            optimizer, scheduler = update_optimizer_for_unfreezing(classification_model, learning_rate=1e-5)
            
        train_loss, train_acc = train_one_epoch(classification_model, loaders["train"], criterion, optimizer, compute_device)
        valid_loss, valid_acc = evaluate_model(classification_model, loaders["valid"], criterion, compute_device)
        
        scheduler.step(valid_loss)
        
        tensorboard_writer.add_scalar("Loss/train", train_loss, epoch)
        tensorboard_writer.add_scalar("Loss/valid", valid_loss, epoch)
        tensorboard_writer.add_scalar("Accuracy/train", train_acc, epoch)
        tensorboard_writer.add_scalar("Accuracy/valid", valid_acc, epoch)
        
        current_lr = optimizer.param_groups[0]['lr']
        tensorboard_writer.add_scalar("Learning_Rate", current_lr, epoch)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.4f} | LR: {current_lr}")
        
        history_train_loss.append(train_loss)
        history_valid_loss.append(valid_loss)
        history_train_acc.append(train_acc)
        history_valid_acc.append(valid_acc)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(classification_model.state_dict(), model_save_path)
            print(f"Validation loss decreased to {best_valid_loss:.4f}. Model saved to '{model_save_path}'.")

    plot_training_metrics(history_train_loss, history_valid_loss, history_train_acc, history_valid_acc)
    tensorboard_writer.close()