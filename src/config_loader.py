import tomllib
from pathlib import Path

class Config:
    def __init__(self, path="config.toml"):
            
        self.path = Path(path)
        self.data = self._load()

    def _load(self):
        
        if not self.path.exists():
            raise FileNotFoundError(f"Config not found: {self.path}")
        with open(self.path, "rb") as f:
            return tomllib.load(f)

    def get(self, *keys):
        
        ref = self.data
        for key in keys:
            ref = ref[key]
        return ref

