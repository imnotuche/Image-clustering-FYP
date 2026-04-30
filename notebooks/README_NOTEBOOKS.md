# Feature Extraction Notebooks Guide

This folder contains Jupyter notebooks for extracting embeddings using **Google Colab** or **Kaggle Notebooks**. Use these when you don't have sufficient GPU resources locally.

## 📋 Notebooks Overview

### 1. `embeddings_extractor.ipynb`

**Purpose**: Extract ResNet-50 embeddings for CIFAR-10

**When to use**:
- Training the DEC pipeline locally (if you extracted embeddings on Colab)
- Quick prototyping with a smaller dataset (160 MB, ~5 minutes to extract)

**What it does**:
- Loads pre-trained ResNet-50 (ImageNet weights)
- Extracts 2048-dimensional normalized embeddings
- Saves to: `cifar10_embeddings.pt` in Google Drive/Kaggle working directory

**Output shape**: (60000, 2048)

**Estimated runtime on Colab GPU**: ~5 minutes

---

### 2. `dino_feature_extractor.ipynb`

**Purpose**: Extract DINO ViT-S/8 embeddings for STL-10 (train + test splits)

**When to use**:
- Training the DINO + projection head pipeline
- Need to cache embeddings before local training (avoids GPU bottleneck)
- Working with limited local VRAM

**What it does**:
- Loads frozen DINO ViT-S/8 from torch.hub
- Extracts 384-dimensional L2-normalized embeddings
- Handles train and test splits separately (preserves indexing)
- Applies standard DINO normalization (matches pipeline defaults)
- Saves two files:
  - `stl10_dino_train_embeddings.pt` — (5000, 384)
  - `stl10_dino_test_embeddings.pt` — (8000, 384)

**Output shapes**: 
- Train: (5000, 384)
- Test: (8000, 384)

**Estimated runtime on Colab GPU**: ~15–20 minutes

**Important**: File names MUST match exactly what the pipeline expects. The pipeline looks for:
```python
'stl10_dino_train_embeddings'
'stl10_dino_test_embeddings'
```

---

### 3. `dino_feature_extractor_tinyimagenet.ipynb`

**Purpose**: Extract DINO ViT-S/8 embeddings for Tiny ImageNet-200

**When to use**:
- Evaluating on a larger, more diverse dataset
- Testing generalization of clustering models
- Extended benchmarking

**What it does**:
- Downloads and extracts Tiny ImageNet-200 (~250 MB)
- Extracts DINO embeddings for train and test splits
- Saves files for reuse

**Output shapes**: 
- Train: (100000, 384)
- Test: (10000, 384)

**Estimated runtime on Colab GPU**: ~30–45 minutes

---

## 🚀 Step-by-Step: Using Colab

### Option A: Upload from GitHub

1. **Open the notebook directly in Colab**
   - Navigate to [Google Colab](https://colab.research.google.com/)
   - File → Open notebook → GitHub
   - Paste: `https://github.com/imnotuche/Image-clustering-FYP`
   - Select desired notebook

2. **Authenticate with Google Drive**
   - Run the "Mount Google Drive" cell
   - Click authorization link
   - Grant permissions

3. **Run all cells** (Ctrl+F9 or Runtime → Run all)
   - Datasets auto-download to `/content/` (temporary)
   - Embeddings save to `My Drive/FYP/` (persistent)

4. **Monitor progress**
   - Each notebook prints extraction progress
   - Check GPU usage in notebook info panel

5. **Download embeddings**
   - Open [Google Drive](https://drive.google.com)
   - Navigate to `FYP/` folder
   - Download `.pt` files (usually 500 MB–1 GB each)

### Option B: Manual Upload

1. Download notebook locally
2. Go to [Google Colab](https://colab.research.google.com/)
3. File → Upload notebook
4. Select downloaded `.ipynb` file
5. Follow steps 2–5 from Option A

---

## 🚀 Step-by-Step: Using Kaggle Notebooks

1. **Go to [Kaggle](https://www.kaggle.com/)**
   - Sign in or create account
   - Create new notebook

2. **Copy-paste notebook code**
   - Open the `.ipynb` file in a text editor (or view on GitHub)
   - Copy code cells
   - Paste into Kaggle notebook cells

3. **Enable GPU**
   - Click notebook options (⚙️)
   - Select "Accelerator" → GPU
   - Apply

4. **Modify save paths** (if needed)
   - Kaggle saves to `/kaggle/working/`
   - Output files are downloadable after execution

5. **Run and download**
   - Execute notebook
   - Download output files from "Output" section

---

## 📁 Organizing Downloaded Embeddings

### After downloading from Colab/Kaggle:

```bash
# Create the embeddings directory structure (if not exists)
mkdir -p embeddings/cifar10
mkdir -p embeddings/stl10
mkdir -p embeddings/tiny-imagenet-200

# Move downloaded .pt files to correct locations
mv ~/Downloads/cifar10_embeddings.pt ./embeddings/cifar10/
mv ~/Downloads/stl10_dino_train_embeddings.pt ./embeddings/stl10/
mv ~/Downloads/stl10_dino_test_embeddings.pt ./embeddings/stl10/
mv ~/Downloads/tinyimagenet_dino_train_embeddings.pt ./embeddings/tiny-imagenet-200/
mv ~/Downloads/tinyimagenet_dino_test_embeddings.pt ./embeddings/tiny-imagenet-200/
```

### Expected structure:
```
embeddings/
├── cifar10/
│   └── cifar10_embeddings.pt
├── stl10/
│   ├── stl10_dino_train_embeddings.pt
│   └── stl10_dino_test_embeddings.pt
└── tiny-imagenet-200/
    ├── tinyimagenet_dino_train_embeddings.pt
    └── tinyimagenet_dino_test_embeddings.pt
```

### Verify downloads:

```bash
# Check file sizes (should be ~500MB–1GB)
ls -lh embeddings/stl10/

# Verify PyTorch files can be loaded
python -c "import torch; data = torch.load('embeddings/stl10/stl10_dino_train_embeddings.pt'); print(data['features'].shape)"
```

---

## ⚡ Performance Tips

### Colab Best Practices

1. **Use high-RAM runtime**
   - Runtime → Change runtime type
   - Select "High RAM" for faster processing

2. **Enable GPU persistence** (Colab Pro only)
   - Prevents timeouts on very long extractions
   - Worth it for Tiny ImageNet

3. **Download before quota resets**
   - Free Colab resets after 12 hours idle
   - Save to Drive immediately after extraction

4. **Split large extractions**
   - If extracting Tiny ImageNet, do train and test separately
   - Reduces memory pressure

### Kaggle Best Practices

1. **Use P100 GPU when available** (faster than K80)
   - Check accelerator type before running

2. **Increase notebook timeout**
   - Settings → Timeout (set to max)
   - Prevents interruption on long extractions

3. **Monitor RAM usage**
   - Use `!nvidia-smi` cell to check GPU memory
   - Reduce batch size if out of memory

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"

**Cause**: PyTorch not available in Colab session

**Solution**: 
```python
# Run this cell in Colab
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

### "No space left on device"

**Cause**: Colab's `/content/` storage is full (~100 GB limit)

**Solution**:
```python
import shutil
# Clear temporary files
shutil.rmtree('/content/stl10', ignore_errors=True)
# Or download to Drive mid-extraction and delete locally
```

---

### "Google Drive quota exceeded"

**Cause**: Drive storage limit (15 GB free tier)

**Solution**:
1. Delete unnecessary files from Drive
2. Or use Kaggle Notebooks instead (more generous storage)
3. Or upload to shared Drive (more space, requires account upgrade)

---

### "Embeddings file can't be loaded"

**Cause**: Corrupted download or wrong data format

**Solution**:
```python
import torch

# Try loading with weights_only=False
data = torch.load('embeddings/stl10/stl10_dino_train_embeddings.pt', weights_only=False)

# Check structure
print(data.keys())  # Should show ['features', 'labels']
print(data['features'].shape)  # Should be (N, embedding_dim)
```

---

### "My embeddings don't match expected shapes"

**Cause**: Different model or normalization used

**Solution**: Verify the notebook uses:
- **ResNet-50**: `models.resnet50(weights=ResNet50_Weights.DEFAULT)`
- **DINO**: `torch.hub.load('facebookresearch/dino:main', 'dino_vits8')`
- **Normalization**: Match exactly to what pipeline expects (printed in notebook)

---

## 🔄 Workflow: Colab → Local

### Complete pipeline from Colab extraction to local training:

```bash
# 1. Run notebook on Colab, save embeddings to Google Drive

# 2. Download embeddings to local machine
mkdir -p embeddings/stl10
# Manually download from Google Drive or:
# wget <drive_share_link> (if publicly shared)

# 3. Verify embeddings loaded correctly
python -c "
import torch
from src.dino_hdbscan.data_manager import DataManager
from torchvision import datasets

manager = DataManager(path='./data/stl10', batch_size=64, device='cuda')
features, labels = manager.load_embedding('stl10_dino_train_embeddings')
print(f'Loaded features: {features.shape}')
print(f'Loaded labels: {labels.shape}')
"

# 4. Run training (embeddings will be loaded from cache, not re-extracted)
python -m dino_hdbscan.run_pipeline

# 5. Test on test set
python -m dino_hdbscan.test_pipeline
```

---

## 📊 Approximate Times & Sizes

| Notebook | Dataset | Extraction Time | Output Size |
|----------|---------|-----------------|-------------|
| embeddings_extractor | CIFAR-10 | 5 min | 500 MB |
| dino_feature_extractor | STL-10 | 20 min | 1.1 GB |
| dino_feature_extractor_tinyimagenet | Tiny ImageNet | 45 min | 1.5 GB |

*Times are for Colab GPU (T4/P100). Local GPU times vary; CPU is 5–10x slower.*

---

## 💡 When to Use Colab vs Local

### Use Colab if:
- ✅ Local GPU VRAM < 8GB (T4 has 16GB)
- ✅ No local CUDA setup
- ✅ First-time feature extraction (no rush)
- ✅ Evaluating on larger datasets (Tiny ImageNet)

### Use Local if:
- ✅ Local GPU VRAM ≥ 8GB
- ✅ Want to iterate quickly (reuse cached embeddings)
- ✅ Prefer not to use external services
- ✅ Need deterministic runs (Colab sessions are ephemeral)

---

## 🔗 Links

- [Google Colab](https://colab.research.google.com/)
- [Kaggle Notebooks](https://www.kaggle.com/code)
- [DINO (torch.hub)](https://github.com/facebookresearch/dino)
- [PyTorch](https://pytorch.org/)
- [torchvision Datasets](https://pytorch.org/vision/stable/datasets.html)

---

**Last Updated**: April 2026

For issues or questions, open a GitHub issue on the main repository.
