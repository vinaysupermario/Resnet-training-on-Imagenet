import torch
import torch.nn as nn
from torchvision import datasets, transforms

import pytest
import glob
import os
import sys

# Import DataSet Class
from train import ImageNetKaggle

# Import Model
from ResNet import Bottleneck, ResNet, ResNet50

def get_latest_model():
    """Get the latest model file from the models directory"""
    model_files = glob.glob('models/model_*.pth')
    if not model_files:
        raise FileNotFoundError("No model files found in models directory. Please add a trained model first.")
    return max(model_files, key=os.path.getctime)

def print_and_log(message):
    """Helper function to ensure message is both printed and logged"""
    print(message, file=sys.stderr)
    return message

def test_input_output_shape():
    """Test 1: Check model's input/output architecture"""
    model = ResNet50(1000)
    test_input = torch.randn(1, 3, 256, 256)
    output = model(test_input)
    assert output.shape == (1, 1000), "Output shape should be (1, 1000)"
    print_and_log(f"\nModel shape test passed: Input (1, 3, 256, 256) -> Output {tuple(output.shape)}")

def test_parameter_count():
    """Test 2: Verify parameter count is less than 30M"""
    model = ResNet50(1000)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 30000000, f"Model has {total_params} parameters, should be less than 30-Million"
    print_and_log(f"\nParameter count test passed: {total_params} parameters")

def test_batch_normalization():
    """Test 3: Check for Batch Normalization layers"""
    model = ResNet50(1000)
    has_batch_norm = any(isinstance(module, nn.BatchNorm2d) for module in model.modules())
    assert has_batch_norm, "Model should use Batch Normalization"
    print_and_log("\nBatch Normalization test passed")

def test_dropout():
    """Test 4: Check for Dropout layers"""
    model = ResNet50(1000)
    has_dropout = any(isinstance(module, nn.Dropout) for module in model.modules())
    assert has_dropout, "Model should use Dropout"
    print_and_log("\nDropout test passed")

def test_gap_or_fc():
    """Test 5: Verify use of GAP or Fully Connected layer"""
    model = ResNet50(1000)
    has_gap = any(isinstance(module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d)) 
                 for module in model.modules())
    has_fc = any(isinstance(module, nn.Linear) for module in model.modules())
    
    assert has_gap or has_fc, "Model should use either Global Average Pooling or Fully Connected Layer"
    print_and_log("\nGAP/FC layer test passed")

def test_epoch_count():
    """Test 6: Verify epoch count is less than 20"""
    from train import EPOCHS
    assert EPOCHS <= 20, f"Epoch count ({EPOCHS}) should be less than or equal to 20"
    print_and_log(f"\nEpoch count test passed: {EPOCHS} epochs")

def test_model_accuracy():
    """Test 7: Check model accuracy on test set (should be > 70%)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_and_log(f"\nUsing device: {device}")

    try:
        # Load the latest model
        model = ResNet50(1000).to(device)
        latest_model = get_latest_model()
        print_and_log(f"Loading model from: {latest_model}")
        
        # Load the model with map_location to handle CPU/GPU differences
        state_dict = torch.load(latest_model, 
                              map_location=device,
                              weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        
        # Load test dataset
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])
    
        # val == validation set
        test_dataset = ImageNetKaggle("/home/shivamkhaneja1/data/imagenet/", "val", transform=test_transforms)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=96, num_workers = 4, shuffle=False)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        print_and_log(f"\nAccuracy test result: {accuracy:.2f}% accuracy")
        assert accuracy > 70, f"Model accuracy is {accuracy:.2f}%, should be > 70%"
    except Exception as e:
        assert False, f"Model couldn't be tested"

if __name__ == "__main__":
    pytest.main(["-v", "--capture=no", __file__]) 