import os
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import torch
from torch.utils.data import Dataset
from datetime import datetime, timezone
from registry_loader import Registry
from config_loader import Config

class DataManager:
    
    def __init__(self, path, batch_size=64, device="cpu"):
        
        if not os.path.isdir(path):
            raise NotADirectoryError(f"Invalid directory: {path}")
        
        p=Path(path)
        
        self.name = p.name
        self.path = path
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
        
        self._load_registry()
        
    def _load_registry(self):
        
        #create config and registry object
        self.config=Config()
        registry_path=f"{self.path}/{self.name}.toml"
        self.registry=Registry(path=registry_path)
        
        #load empty dicts in the registry file if it dosent exist
        if not os.path.exists(registry_path):
            embeddings={}
            models={}
            self.registry.register("embeddings", embeddings)
            self.registry.register("models", models)

    def get_loader(self, source, train=True, shuffle=True):
        
        """
        source: 'local' for custom class OR a torch dataset class (e.g. datasets.CIFAR10)
        """
        
        if source == 'local':
            print(f"Loading local images from: {self.path}...")
            dataset = UnlabeledImageDataset(root_dir=self.path, transform=self.transform)
        
        else:
            # Dynamically initialize any torchvision dataset (CIFAR10, CIFAR100, STL10, etc.)
            print(f"Loading {source.__name__} from: {self.path}...")
            dataset = source(
                root=self.path, 
                train=train, 
                download=True, 
                transform=self.transform
            )
        
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=2)  
    
    def store_embedding(self, embeddings, name=f"embedding-{datetime.now(timezone.utc).strftime("%d-%m-%Y_%H-%M-%S")}", overwrite=False):
        #path reserved just for embeddings
        embeddings_root=self.config.get("paths", "embeddings_dir")
        
        #save path of the embedding file
        save_path=f"./{embeddings_root}/{self.name}/{name}.pt"
        
        #create parent directory if it dosent exist
        path_obj = Path(save_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        #rename if filename already exists
        path_exists=os.path.exists(save_path)
        while path_exists and overwrite is False:
            count=1
            save_path=f"./{embeddings_root}/{self.name}/{name}_{count}.pt"
            name=f"{name}_{count}"
            count=count+1
            path_exists=os.path.exists(save_path)
        
        torch.save(embeddings, save_path)
        print(f"Saved embeddings at {save_path}.")
        
        #update registry
        data={
            f"{name}": save_path
        }
        self.registry.register("embeddings", data)
        print(f"{self.name} registry updated.")
        
        
    def store_model(self, model, name=f"model-{datetime.now(timezone.utc).strftime("%d-%m-%Y_%H-%M-%S")}", overwrite=False):
        #path reserved just for models
        models_root=self.config.get("paths", "models_dir")
        
        #save path of the model file
        save_path=f"./{models_root}/{self.name}/{name}.pth"
        
        #create parent directory if it dosent exist
        path_obj = Path(save_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        #rename if filename already exists
        path_exists=os.path.exists(save_path)
        while path_exists and overwrite is False:
            count=1
            save_path=f"./{models_root}/{self.name}/{name}_{count}.pth"
            name=f"{name}_{count}"
            count=count+1
            path_exists=os.path.exists(save_path)
        
        torch.save(model, save_path)
        print(f"Saved model at {save_path}.")
        
        #update registry
        data={
            f"{name}": save_path
        }
        self.registry.register("models", data)
        print(f"{self.name} registry updated.")
        
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
    
test=DataManager("./data/cifar10")
test.store_model({"test":"someshi"})
test.store_model({"teswwt":"somewweeshi"}, overwrite=True)