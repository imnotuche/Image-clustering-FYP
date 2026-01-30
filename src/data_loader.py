'''
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(batch_size=64, train=True, shuffle=False):
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
    dataset = datasets.CIFAR10(
        root='../data/cifar10', 
        train=train, 
        download=True, 
        transform=transform
    )
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return loader
'''

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# This is the "Flat Folder" loader for your frontend uploads
class UnlabeledImageDataset(Dataset):
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Only grab files that are actually images
        self.image_files = [
            f for f in os.listdir(root_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0  

def get_dataloader(batch_size=64, train=True, shuffle=False, mode='dataset', local_path=None):
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    if mode == 'dataset':
        print("Loading CIFAR-10 Dataset...")
        dataset = datasets.CIFAR10(
            root='../data/cifar10', 
            train=train, 
            download=True, 
            transform=transform
        )
    
    elif mode == 'local':
        if local_path is None or not os.path.exists(local_path):
            raise ValueError("Invalid local_path.")
        
        print(f"Loading uploaded images from: {local_path}")
        # Use our custom class instead of ImageFolder
        dataset = UnlabeledImageDataset(root_dir=local_path, transform=transform)
    
    else:
        raise ValueError("Invalid mode. Choose 'dataset' or 'local'.")
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader