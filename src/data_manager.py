import os
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import Dataset
from datetime import datetime, timezone
from registry_loader import Registry
from config_loader import Config

class DataManager:
    
    def __init__(self, path, batch_size=64, device="cpu"):
        
        #check if provided path exixts
        if not os.path.isdir(path):
            raise NotADirectoryError(f"Invalid directory: {path}")
        
        p=Path(path)
        
        self.name = p.name
        self.path = path
        self.batch_size = batch_size
        self.device = device
        
        # Standard DINO normalization
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])
        
        self._verify_root_directories()
        self._load_registry()
        
    def _verify_root_directories(self):
        """
        this makes sure all the directories needed by the class exist
        """
        
        self.config=Config() #config loader object
        self.config.get("paths") #get all paths
        root_directories=list(self.config.get("paths").values())
        
        for path in root_directories:
            if not os.path.exists(f"{path}/{self.name}"):
                os.mkdir(f"./{path}/{self.name}")
                print(f"Created missing directory: {path}/{self.name}")

    def _load_registry(self):
        """
        this loads and initialises the registry config file for
        storing metadata
        """
        
        #create object
        registry_path=f"{self.path}/{self.name}.toml"
        self.registry=Registry(path=registry_path)
        
        #load empty dicts in the registry file if it dosent exist
        if not os.path.exists(registry_path):
            embeddings={}
            models={}
            self.registry.register("embeddings", embeddings)
            self.registry.register("models", models)

    def update_registry(self):
        
        """
        This updates registry to match what actually exists
        run this in a separate script to update registry if you delete a file
        or manually add one
        """
        
        #retrieve embeddings root path and paths of embeddings from registry
        embeddings_root=f"{self.config.get("paths" ,"embeddings_dir")}/{self.name}"
        embeddings_registry=list(self.registry.get("embeddings").values())
        
        #get the files present in the embeddings folder
        embeddings_files=os.listdir(embeddings_root)
        
        #register new files
        print(f"Checking for unregistered embedding files for {self.name}")
        for file in embeddings_files:
            
            #assemble fullpath to compare with paths in registry
            full_path=f"./{embeddings_root}/{file}"
            #split the extension from the filename
            name, _=os.path.splitext(file)
            
            #save to registry if no record exists
            if full_path not in embeddings_registry:
                data={
                    name: full_path
                }
                self.registry.register("embeddings", data)
                print(f"Added {file} to registry")
        
        #Remove records of deleted files
        print(f"Checking for Records of deleted embedding files for {self.name}")
        for path in embeddings_registry:
            
            p=Path(path)
            if p.name not in embeddings_files:
                #split the extension from the filename
                name=p.stem
                self.registry.unregister("embeddings", name)
                print(f"Removed {p.name} record from registry")
        
        #retrieve models root path and paths of models from registry
        models_root=f"{self.config.get("paths" ,"models_dir")}/{self.name}"
        models_registry=list(self.registry.get("models").values())
        
        #get the files present in the embeddings folder
        models_files=os.listdir(models_root)
        
        #register new files
        print(f"Checking for unregistered models for {self.name}")
        for file in models_files:
            
            #assemble fullpath to compare with paths in registry
            full_path=f"./{models_root}/{file}"
            #split the extension from the filename
            name, _=os.path.splitext(file)
            
            #save to registry if no record exists
            if full_path not in models_registry:
                data={
                    name: full_path
                }
                self.registry.register("models", data)
                print(f"Added {file} to registry")
                
        #Remove records of deleted files
        print(f"Checking for Records of deleted model files for {self.name}")
        for path in models_registry:
            
            p=Path(path)
            if p.name not in models_files:
                #split the extension from the filename
                name=p.stem
                self.registry.unregister("models", name)
                print(f"Removed {p.name} record from registry")
        

    def get_loader(self, source, train=True, shuffle=True):
        
        """
        source: 'local' for custom class OR a torch dataset class (e.g. datasets.CIFAR10)
        """
        
        if source == 'local':
            print(f"Loading local images from: {self.path}...")
            dataset = UnlabeledImageDataset(root_dir=self.path, transform=self.transform)
        
        else:
            # Dynamically initialize any torchvision dataset (CIFAR10, CIFAR100, STL10, etc.)
            try:
                # CIFAR-10, CIFAR-100 use train=True/False
                dataset = source(
                    root=self.path,
                    train=train,
                    download=True,
                    transform=self.transform
                )
            except TypeError:
                # STL-10 and others use split= instead of train=
                split = 'train' if train else 'test'
                dataset = source(
                    root=self.path,
                    split=split,
                    download=True,
                    transform=self.transform
                )
        
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=0)  
    
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
        count=1
        while path_exists and overwrite is False:
            save_path=f"./{embeddings_root}/{self.name}/{name}_{count}.pt"
            name=f"{name}_{count}"
            count=count+1
            path_exists=os.path.exists(save_path)
        
        torch.save(embeddings, save_path)
        print(f"Saved {name} at {save_path}.")
        
        #update registry
        data={
            f"{name}": save_path
        }
        self.registry.register("embeddings", data)
        print(f"{self.name} registry updated.")
        
    def load_embedding(self, name):
        path=self.registry.get("embeddings", name)
        data = torch.load(path, weights_only=False)
        return data["features"], data["labels"]
        
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
        count=1
        while path_exists and overwrite is False:
            save_path=f"./{models_root}/{self.name}/{name}_{count}.pth"
            name=f"{name}_{count}"
            count=count+1
            path_exists=os.path.exists(save_path)
        
        torch.save(model, save_path)
        print(f"Saved {name} at {save_path}.")
        
        #update registry
        data={
            f"{name}": save_path
        }
        self.registry.register("models", data)
        print(f"{self.name} registry updated.")
        
    def load_model(self, name, device):
        path = self.registry.get("models", name)
        return torch.load(path, map_location=device, weights_only=False)
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
    
#test=DataManager("./data/stl10")
#test.store_embedding({"test":"someshi"})
#test.update_registry()
#test.store_model({"teswwt":"somewweeshi"}, overwrite=True)

