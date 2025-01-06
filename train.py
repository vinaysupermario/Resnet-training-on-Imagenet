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
from pathlib import Path

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
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            
            # Top-1 accuracy
            _, pred = output.topk(1, 1, True, True)
            correct_top1 += pred.eq(target.view(-1, 1)).sum().item()
            
            # Top-5 accuracy
            _, pred = output.topk(5, 1, True, True)
            target_reshaped = target.view(-1, 1).expand_as(pred)
            correct_top5 += pred.eq(target_reshaped).sum().item()
            
            total += target.size(0)

    test_loss /= len(test_loader)
    top1_accuracy = 100. * correct_top1 / total
    top5_accuracy = 100. * correct_top5 / total

    print(f'\nTest set: Average loss: {test_loss:.4f}')
    print(f'Top-1 Accuracy: {correct_top1}/{total} ({top1_accuracy:.2f}%)')
    print(f'Top-5 Accuracy: {correct_top5}/{total} ({top5_accuracy:.2f}%)\n')
    
    return test_loss, top1_accuracy, top5_accuracy

def save_checkpoint(epoch, model, optimizer, scheduler, train_losses, test_losses, 
                   train_acc, test_top1_acc, test_top5_acc, best_acc, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_acc': train_acc,
        'test_top1_acc': test_top1_acc,
        'test_top5_acc': test_top5_acc,
        'best_acc': best_acc,
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return (checkpoint['epoch'], checkpoint['train_losses'], checkpoint['test_losses'],
            checkpoint['train_acc'], checkpoint['test_top1_acc'], 
            checkpoint['test_top5_acc'], checkpoint['best_acc'])

def run() -> Tuple[torch.nn.Module, str]:
    # Create necessary directories
    Path('models').mkdir(exist_ok=True)
    Path('checkpoints').mkdir(exist_ok=True)
    Path('logs').mkdir(exist_ok=True)

    # Initialize metrics storage
    train_losses = []
    test_losses = []
    train_acc = []
    test_top1_acc = []
    test_top5_acc = []
    best_acc = 0.0
    patience = 10
    patience_counter = 0
    start_epoch = 0

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
    pred = model(torch.rand(1,3,256,256).to(device))
    # print(model)
    # summary(model, input_size=(3, 256, 256))

## 8. Train and Test the Model

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # Load checkpoint if exists
    checkpoint_path = Path('checkpoints/latest_checkpoint.pth')
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        start_epoch, train_losses, test_losses, train_acc, test_top1_acc, test_top5_acc, best_acc = load_checkpoint(
            model, optimizer, scheduler, checkpoint_path
        )
        print(f"Resuming from epoch {start_epoch+1}")

    # Training loop
    start_time = time.time()
    
    for epoch in range(start_epoch, EPOCHS):
        epoch_start_time = time.time()
        print(f"\nEPOCH: {epoch+1}/{EPOCHS}")
        
        train_loss, train_accuracy = train_model(model, device, train_loader, criterion, optimizer, epoch)
        test_loss, test_top1_accuracy, test_top5_accuracy = test_model(model, device, test_loader, criterion)
        
        epoch_time = time.time() - epoch_start_time
        print(f'Epoch time elapsed: {epoch_time/60:.2f} minutes')
        
        # Record metrics
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_acc.append(train_accuracy)
        test_top1_acc.append(test_top1_accuracy)
        test_top5_acc.append(test_top5_accuracy)
        
        # Step the scheduler
        scheduler.step(test_loss)
        
        # Save regular checkpoint
        save_checkpoint(
            epoch, model, optimizer, scheduler,
            train_losses, test_losses, train_acc, 
            test_top1_acc, test_top5_acc, best_acc, 
            'checkpoints/latest_checkpoint.pth'
        )
        
        # Save best model (using top-1 accuracy as criterion)
        if test_top1_accuracy > best_acc:
            best_acc = test_top1_accuracy
            patience_counter = 0
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            best_model_path = f'models/best_model_{timestamp}_top1acc{test_top1_accuracy:.2f}.pth'
            save_checkpoint(
                epoch, model, optimizer, scheduler,
                train_losses, test_losses, train_acc,
                test_top1_acc, test_top5_acc, best_acc,
                best_model_path
            )
            print(f'Saved best model with Top-1 accuracy: {test_top1_accuracy:.2f}%')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    total_time = time.time() - start_time
    print(f'\nTotal training completed in {total_time/60:.2f} minutes')

    # Save training history with top-5 accuracy
    history = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_acc': train_acc,
        'test_top1_acc': test_top1_acc,
        'test_top5_acc': test_top5_acc,
        'training_time': total_time
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'logs/training_history_{timestamp}.json', 'w') as f:
        json.dump(history, f)

    # Update plotting to include top-5 accuracy
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 4, 2)
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(test_top1_acc, label='Test Top-1 Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.subplot(1, 4, 3)
    plt.plot(test_top5_acc, label='Test Top-5 Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.subplot(1, 4, 4)
    plt.plot(lrs)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    
    plt.tight_layout()
    plt.savefig(f'logs/training_history_{timestamp}.png')
    
    return model, best_model_path

EPOCHS = 1
if __name__ == "__main__":
    run()