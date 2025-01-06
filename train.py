## 1. Import the Libraries

import gc
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os
import time
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import logging
from typing import Tuple

# Import PyTorch libraries
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
from torch.optim.lr_scheduler import StepLR

# Import Model
from ResNet import Bottleneck, ResNet, ResNet50

print("Libraries imported - ready to use PyTorch", torch.__version__)

def show_image(image, label):
    image = image.permute(1, 2, 0)
    plt.imshow(image.squeeze())
    plt.title(f'Label: {label}')
    plt.show()

## 2. Define the DataSet Class

class ImageNetKaggle(Dataset):
    def __init__(self, root, split, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
                    json_file = json.load(f)
                    for class_id, v in json_file.items():
                        self.syn_to_class[v[0]] = int(class_id)
        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
                    self.val_to_syn = json.load(f)
        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
        for entry in os.listdir(samples_dir):
            if split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        x = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, self.targets[idx]

## 9. Training and Testing Functions for model

def train_model(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    running_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        y_pred = model(data)

        loss = criterion(y_pred, target)  # Using the defined criterion
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc= f'Loss={loss.item():.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    
    return running_loss / len(train_loader), 100. * correct / processed

def test_model(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return test_loss, accuracy

def run() -> Tuple[torch.nn.Module, str]:
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []
    best_acc = 0.0
    patience = 10
    patience_counter = 0

## 3. Initialize the SEED
    
    SEED = 1
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)
    torch.manual_seed(SEED)
    if cuda:
        torch.cuda.manual_seed(SEED)

## 4. Clear out the NVIDIA RAM

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.memory_summary(device=None, abbreviated=False)

## 5. Define the Data Transforms

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    # Train Phase transformations
    train_transforms = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean,std)])
    
    # Validation Phase transformations
    test_transforms = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean,std)])

## 6. Import the Data using Dataset, and Define the DataLoader to create batches

    train = ImageNetKaggle("/home/shivamkhaneja1/data/imagenet/", "train", transform=train_transforms)
    test = ImageNetKaggle("/home/shivamkhaneja1/data/imagenet/", "val", transform=test_transforms)

    dataloader_args = dict(shuffle=True, batch_size=96, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
    
    train_loader = torch.utils.data.DataLoader(train, **dataloader_args)
    test_loader = torch.utils.data.DataLoader(test, **dataloader_args)

## 7. Create the Model
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = ResNet50(len(train.syn_to_class)).to(device)
    # print(model)
    # summary(model, input_size=(3, 256, 256))

## 8. Train and Test the Model

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience=5)

    for epoch in range(EPOCHS):
        print(f"EPOCH: {epoch+1}/{EPOCHS}")
        
        train_loss, train_accuracy = train_model(model, device, train_loader, criterion, optimizer, epoch)
        test_loss, test_accuracy = test_model(model, device, test_loader, criterion)
        
        # Record metrics
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_acc.append(train_accuracy)
        test_acc.append(test_accuracy)
        
        # Step the scheduler
        scheduler.step(test_loss)
        
        # Early stopping
        if test_accuracy > best_acc:
            best_acc = test_accuracy
            patience_counter = 0
            # Save best model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f'models/model_{timestamp}_acc{test_accuracy:.2f}.pth'
            torch.save(model.cpu().state_dict(), save_path)
            model.to(device)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(test_acc, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'training_history_{timestamp}.png')
    
    return model, save_path

EPOCHS = 1
if __name__ == "__main__":
    run()