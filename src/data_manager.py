import os
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset


class UnlabeledImageDataset(Dataset):
    
    def __init__(self, root_dir, transform=None):
        
        if not os.path.isdir(root_dir):
            raise NotADirectoryError(f"Invalid directory: {root_dir}")
        
        self.root_dir = root_dir
        self.transform = transform
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
        # Return 0 as a placeholder for the label to maintain (image, label) format
        return image, 0


class DataManager:
    
    def __init__(self, batch_size=64, device="cpu"):
        
        self.batch_size = batch_size
        self.device = device
        # Standard ResNet normalization
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

    def get_loader(self, source, path, train=True, shuffle=True):
        
        """
        source: 'local' for custom class OR a torch dataset class (e.g. datasets.CIFAR10)
        path: root directory of the data
        """
        
        if not os.path.isdir(path):
            raise NotADirectoryError(f"Invalid directory: {path}")
        
        if source == 'local':
            print(f"Loading local images from: {path}...")
            dataset = UnlabeledImageDataset(root_dir=path, transform=self.transform)
        
        else:
            # Dynamically initialize any torchvision dataset (CIFAR10, CIFAR100, STL10, etc.)
            print(f"Loading {source.__name__} from: {path}...")
            dataset = source(
                root=path, 
                train=train, 
                download=True, 
                transform=self.transform
            )
        
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=2)  