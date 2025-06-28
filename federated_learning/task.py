from collections import OrderedDict
import os
import pickle
import numpy as np
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

from flwr.common import NDArrays

# CIFAR-10 class names
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"
]

# Defines the CIFAR-10 dataset.
class CIFAR10Dataset(Dataset):
    
    # Initializes the dataset.
    def __init__(self, data_dir: str, train: bool = True, transform=None):
        self.transform = transform
        self.data = []
        self.targets = []
        
        if train:
            # Load training batches
            for i in range(1, 6):
                batch_file = os.path.join(data_dir, f"data_batch_{i}")
                with open(batch_file, 'rb') as f:
                    batch = pickle.load(f, encoding='bytes')
                    self.data.append(batch[b'data'])
                    self.targets.extend(batch[b'labels'])
            self.data = np.vstack(self.data)
        else:
            # Load test batch
            test_file = os.path.join(data_dir, "test_batch")
            with open(test_file, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
                self.data = batch[b'data']
                self.targets = batch[b'labels']
        
        self.data = self.data.reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))

    # Returns the length of the dataset.
    def __len__(self):
        return len(self.data)

    # Returns a sample from the dataset.
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        
        # Convert to PIL Image for transforms
        img = img.astype(np.uint8)
        
        if self.transform:
            img = self.transform(img)
        
        return img, target

# Returns the data transformations for the CIFAR-10 dataset.
def get_cifar10_transforms(train: bool = True):
    if train:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    return transform

# Loads and partitions the CIFAR-10 dataset.
def load_data(partition_id: int, num_partitions: int, batch_size: int, data_dir: str = "cifar-10-batches-py"):
    
    # Load full dataset
    train_transform = get_cifar10_transforms(train=True)
    test_transform = get_cifar10_transforms(train=False)
    
    train_dataset = CIFAR10Dataset(data_dir, train=True, transform=train_transform)
    test_dataset = CIFAR10Dataset(data_dir, train=False, transform=test_transform)
    
    # Partition training data
    total_train_samples = len(train_dataset)
    samples_per_partition = total_train_samples // num_partitions
    
    start_idx = partition_id * samples_per_partition
    if partition_id == num_partitions - 1:
        end_idx = total_train_samples  # Last partition gets remaining samples
    else:
        end_idx = start_idx + samples_per_partition
    
    # Create subset for this partition
    train_indices = list(range(start_idx, end_idx))
    train_subset = Subset(train_dataset, train_indices)
    
    # For validation, use a small portion of test set (same for all clients)
    val_size = len(test_dataset) // 4  # Use 25% of test set for validation
    val_indices = list(range(val_size))
    val_subset = Subset(test_dataset, val_indices)
    
    # Create data loaders
    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Client {partition_id}: {len(train_subset)} train samples, {len(val_subset)} val samples")
    
    return trainloader, valloader

# Returns the ResNet-18 model.
def get_model():
    model=resnet18(pretrained=False)
    #model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # Modify first conv layer for CIFAR-10 (32x32 images)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove maxpool for small images
    # Modify final layer for 10 classes
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    return model

# Trains the model on the provided training set.
def train(net: nn.Module, trainloader: DataLoader, epochs: int, learning_rate: float, device: torch.device):
    if len(trainloader.dataset) == 0:
        print("Training dataloader is empty. Skipping training.")
        return 0.0

    net.to(device)
    net.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    total_loss = 0.0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_samples = 0
        
        for batch_idx, (data, target) in enumerate(trainloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * data.size(0)
            epoch_samples += data.size(0)
        
        avg_epoch_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0.0
        total_loss += avg_epoch_loss
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_epoch_loss:.4f}")
    
    return total_loss / epochs if epochs > 0 else 0.0

# Tests the model on the provided test set.
def test(net: nn.Module, testloader: DataLoader, device: torch.device):
    if len(testloader.dataset) == 0:
        print("Test dataloader is empty. Skipping testing.")
        return 0.0, 0.0, [], []

    net.to(device)
    net.eval()
    
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for data, target in testloader:
            y_true.extend(target.tolist())
            data, target = data.to(device), target.to(device)
            output = net(data)
            
            # Calculate loss
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            
            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            y_pred.extend(pred.cpu().numpy().flatten().tolist())
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += data.size(0)
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = correct / total_samples if total_samples > 0 else 0.0
    
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f} ({correct}/{total_samples})")
    
    return avg_loss, accuracy, y_true, y_pred

# Extracts the model weights.
def get_weights(net: nn.Module) -> NDArrays:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

# Sets the model weights.
def set_weights(net: nn.Module, parameters: NDArrays) -> None:
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

if __name__ == "__main__":
    # Test the data loading
    print("Testing CIFAR-10 data loading...")
    
    data_dir = "../cifar-10-batches-py"
    if os.path.exists(data_dir):
        trainloader, valloader = load_data(
            partition_id=0, 
            num_partitions=2, 
            batch_size=32, 
            data_dir=data_dir
        )
        
        print(f"Train batches: {len(trainloader)}")
        print(f"Val batches: {len(valloader)}")
        
        # Test model
        model = get_model()
        print(f"Model created: {model.__class__.__name__}")
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
        
        # Test one batch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        for batch_idx, (data, target) in enumerate(trainloader):
            print(f"Batch {batch_idx}: Data shape: {data.shape}, Target shape: {target.shape}")
            data = data.to(device)
            output = model(data)
            print(f"Output shape: {output.shape}")
            break
    else:
        print(f"CIFAR-10 data directory not found at {data_dir}") 