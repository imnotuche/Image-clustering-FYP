import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(batch_size=64):
    """
    Prepares and returns the CIFAR-10 data loader.
    Includes resizing and normalization for ResNet-50 compatibility.
    """
    # ResNet-50 expects 224x224 images and specific normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # ImageNet standards
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load the CIFAR-10 dataset
    # 'train=False' loads the test set (10,000 images), which is great for testing
    dataset = datasets.CIFAR10(
        root='../data/cifar10', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return loader