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
        self.data[name] = data
        self._save()

    def get(self, *keys):   
        ref = self.data
        for key in keys:
            ref = ref[key]
        return ref

    def _save(self):
        with open(self.path, "wb") as f:
            f.write(tomli_w.dumps(self.data).encode())