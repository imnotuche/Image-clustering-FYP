import tomllib
import tomli_w
from pathlib import Path

class Registry:
    def __init__(self, path):
        self.path = Path(path)
        self.data = self._load()

    def _load(self):
        if not self.path.exists():
            return {}
        with open(self.path, "rb") as f:
            return tomllib.load(f)

    def register(self, name, data):
        if name in self.data and isinstance(self.data[name], dict) and isinstance(data, dict):
            self.data[name].update(data)
        else:
            self.data[name] = data 
        self._save()
        
    def unregister(self, *keys):
        if not keys:
            return

        # Navigate to the parent dictionary of the target key
        ref = self.data
        for key in keys[:-1]:
            ref = ref[key]
        
        # Remove the specific key
        target_key = keys[-1]
        if target_key in ref:
            del ref[target_key]
            self._save()

    def get(self, *keys):   
        ref = self.data
        for key in keys:
            ref = ref[key]
        return ref

    def _save(self):
        with open(self.path, "wb") as f:
            f.write(tomli_w.dumps(self.data).encode())