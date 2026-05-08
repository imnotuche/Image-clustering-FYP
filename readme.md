# Image Clustering System

An unsupervised image clustering framework leveraging pretrained convolutional neural network embeddings (ResNet-50 and DINO ViT) combined with clustering algorithms (Deep Embedded Clustering, HDBSCAN, UMAP) to group visually similar images without labeled supervision. The repository implements three distinct clustering pipelines with different embedding strategies and architecture choices.

## Key Features

- **Multiple embedding models**: ResNet-50 and frozen DINO ViT-S/8 feature extraction
- **Flexible clustering approaches**: Deep Embedded Clustering (DEC), HDBSCAN, UMAP dimensionality reduction
- **Pseudo-label generation**: Automatic pseudo-label creation with HDBSCAN and noise reassignment
- **Contrastive learning**: Supervised contrastive loss with trainable projection heads
- **Desktop UI**: Interactive graphical interface for DINO-based clustering (PySide6/Qt)
- **Comprehensive evaluation**: NMI, ARI, Silhouette Score, and visualization utilities
- **Multi-dataset support**: CIFAR-10, STL-10, Tiny ImageNet-200, and custom image folders

## Tech Stack

**Core Framework**
- PyTorch 2.9.1 — Deep learning
- PyTorch Lightning 2.6.1 — Training orchestration
- torchvision 0.24.1 — Pretrained models and datasets

**Feature Extraction & Clustering**
- DINO ViT-S/8 (via torch.hub)
- scikit-learn — HDBSCAN, PCA, evaluation metrics
- UMAP 0.5.11 — Dimensionality reduction
- Lightly 1.5.22 — Self-supervised learning utilities

**UI & Visualization**
- PySide6 6.11.0 — Desktop application framework
- Matplotlib 3.10.8 — Plotting and galleries
- NumPy 2.3.5, Pandas 3.0.0 — Data manipulation

**Configuration & Utilities**
- Hydra Core 1.3.2 — Configuration management
- PyYAML 6.0.3 — Config file parsing
- Pydantic 2.12.5 — Data validation
- Joblib 1.5.3 — Model serialization
- tqdm 4.67.1 — Progress tracking

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (optional, for GPU acceleration)
- 8GB+ RAM (16GB+ recommended for large datasets)
- ~3GB free disk space (for cached datasets and embeddings)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/imnotuche/Image-clustering-FYP.git
   cd Image-clustering-FYP
   ```

2. **Create and activate a virtual environment**
   ```bash
   # Using venv
   python -m venv venv
   source venv/Scripts/activate  # On Windows: venv\Scripts\activate
   
   # Or using conda
   conda create -n clustering python=3.9
   conda activate clustering
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare data directories**
   ```bash
   mkdir -p data embeddings models results
   ```
   > The application will create subdirectories automatically upon first run. Datasets are auto-downloaded on first run.

## Project Structure

```
Image-clustering-FYP/
├── src/
│   ├── DEC-pipeline/              # Deep Embedded Clustering pipeline
│   │   ├── main.py                # Training entry point (CIFAR-10)
│   │   ├── test_cluster.py        # Evaluation on test set
│   │   ├── train.py               # DEC model training loop
│   │   ├── feature_extractor.py   # ResNet-50 embeddings
│   │   ├── dimension_reduction.py # PCA reduction
│   │   ├── dec_model.py           # Deep Embedded Clustering architecture
│   │   └── utils.py               # Metrics & visualization
│   │
│   ├── dino_hdbscan/              # DINO + HDBSCAN + Projection Head pipeline
│   │   ├── main.py                # Desktop UI launcher
│   │   ├── run_pipeline.py        # Training pipeline entry (STL-10)
│   │   ├── test_pipeline.py       # Evaluation on test set
│   │   ├── run_custom.py          # Cluster custom image folders
│   │   ├── feature_extractor.py   # DINO ViT-S/8 frozen backbone
│   │   ├── embedding_model.py     # Trainable projection head (384→128d)
│   │   ├── train.py               # Projection head training
│   │   ├── dimension_reduction.py # UMAP dimensionality reduction
│   │   ├── pseudo_label_generator.py # HDBSCAN + noise reassignment
│   │   ├── inference.py           # Inference pipeline
│   │   ├── evaluate.py            # Metrics & visualization
│   │   ├── auto_min_cluster.py    # Dynamic min_cluster_size
│   │   └── UI/                    # PySide6 desktop application
│   │       ├── dino_hdbscan_app.py
│   │       ├── components/        # UI widgets
│   │       ├── workers/           # Background threads
│   │       └── styles.qss         # Qt stylesheets
│   │
│   ├── hdbscan_pipeline/          # UMAP + HDBSCAN pipeline
│   │   ├── main.py                # Training entry point (STL-10)
│   │   ├── test_cluster.py        # Evaluation on test set
│   │   ├── train.py               # HDBSCAN + UMAP training
│   │   ├── feature_extractor.py   # ResNet-50 embeddings
│   │   ├── dimension_reduction.py # UMAP reduction (→2D)
│   │   ├── inference.py           # Clustering inference
│   │   └── utils.py               # Metrics & visualization
│   │
│   └── (shared modules across pipelines)
│       ├── data_manager.py        # Dataset loading & caching
│       ├── config_loader.py       # Configuration parsing
│       └── registry_loader.py     # Model/embedding registry
│
├── data/
│   ├── cifar10/                   # CIFAR-10 dataset
│   ├── stl10/                     # STL-10 dataset
│   ├── tiny-imagenet-200/         # Tiny ImageNet dataset
│   └── high-res-test/             # Custom high-res test images
│
├── embeddings/
│   ├── cifar10/                   # Cached CIFAR-10 embeddings
│   ├── stl10/                     # Cached STL-10 embeddings
│   └── tiny-imagenet-200/         # Cached Tiny ImageNet embeddings
│
├── models/
│   ├── cifar10/                   # DEC model weights
│   ├── stl10/                     # DINO projector & UMAP models
│   └── tiny-imagenet-200/         # Additional model variants
│
├── results/
│   ├── experiment_results.pt      # Metrics from training
│   ├── galleries_*/               # Cluster visualization galleries
│   └── *.pt                       # Raw prediction tensors
│
├── notebooks/
│   ├── embeddings_extractor.ipynb              # ResNet-50 feature extraction (Colab/Kaggle)
│   ├── dino_feature_extractor.ipynb            # DINO ViT-S/8 extraction for STL-10 (Colab)
│   ├── dino_feature_extractor_tinyimagenet.ipynb # DINO extraction for Tiny ImageNet (Colab)
│   └── README_NOTEBOOKS.md                     # Instructions for Colab/Kaggle feature extraction
│
├── config.toml                    # Base configuration
├── pyproject.toml                 # Project metadata
├── requirements.txt               # Python dependencies
└── LICENSE                        # MIT License
```

## Notebooks: Feature Extraction on Colab/Kaggle

If you don't have local GPU resources, the provided Jupyter notebooks allow you to extract embeddings using **Google Colab** or **Kaggle Notebooks** for free.

### Available Notebooks

| Notebook | Dataset | Embedding Model | Output | Colab-Ready |
|----------|---------|-----------------|--------|------------|
| `embeddings_extractor.ipynb` | CIFAR-10 | ResNet-50 (2048-dim) | `cifar10_embeddings.pt` | ✓ |
| `dino_feature_extractor.ipynb` | STL-10 | DINO ViT-S/8 (384-dim) | `stl10_dino_train/test_embeddings.pt` | ✓ |
| `dino_feature_extractor_tinyimagenet.ipynb` | Tiny ImageNet | DINO ViT-S/8 (384-dim) | `tinyimagenet_dino_train/test_embeddings.pt` | ✓ |

### Quick Start: Running on Google Colab

1. **Copy notebook to Colab**
   - Open the notebook file from this repository
   - Click "Open in Colab" (or manually upload to colab.research.google.com)

2. **Mount Google Drive** (for persistent storage)
   - The notebooks include a mount cell; execute it and authorize
   - All embeddings will be saved to `Google Drive/FYP/` folder

3. **Run all cells**
   - Each notebook is self-contained with all necessary imports
   - Datasets auto-download to Colab's `/content/` temporary storage
   - Embeddings save to Google Drive automatically

4. **Download embeddings locally**
   - Go to `My Drive/FYP/` in Google Drive
   - Download the `.pt` files (usually 500MB–1GB each)
   - Place in `embeddings/<dataset>/` in your local repo:
     ```bash
     # After downloading from Drive:
     mv ~/Downloads/stl10_dino_train_embeddings.pt ./embeddings/stl10/
     mv ~/Downloads/stl10_dino_test_embeddings.pt ./embeddings/stl10/
     ```

5. **Run pipelines locally**
   - With embeddings cached, training is much faster (no re-extraction needed)
   ```bash
   python -m dino_hdbscan.run_pipeline  # Will load cached embeddings instead
   ```

### Running on Kaggle Notebooks

Kaggle Notebooks also provide free GPU and have slightly more storage. Steps are similar:

1. Upload notebook to Kaggle (or search for existing public versions)
2. Select **GPU accelerator** from settings
3. Update save path from Google Drive to Kaggle's `/kaggle/working/` output folder
4. Download embeddings from Kaggle's output folder after execution

### Why Use Colab/Kaggle?

- **Free GPU**: Avoid local VRAM limitations (RTX 4090 equivalent, 16GB)
- **No installation**: Pre-configured Python + PyTorch (no dependency hell)
- **Large dataset support**: Colab caches are temporary but sufficient for extraction
- **Persistent storage**: Save embeddings to Google Drive for reuse locally
- **Time-saving**: DINO extraction on GPU takes ~10–20 minutes (vs. hours on CPU)

### Notebook Details

**`embeddings_extractor.ipynb`** (ResNet-50, CIFAR-10):
- Designed for quick prototyping
- Downloads and extracts CIFAR-10 (160 MB) in ~5 minutes
- Saves: `cifar10_embeddings.pt` (60,000 images × 2048-dim)
- Use for: DEC pipeline training

**`dino_feature_extractor.ipynb`** (DINO, STL-10):
- Extracts train AND test splits separately (ensures correct indexing)
- Downloads STL-10 (~2.3 GB) on first run
- Saves: `stl10_dino_train_embeddings.pt`, `stl10_dino_test_embeddings.pt`
- Uses **standard DINO normalization** (matches pipeline normalization)
- Use for: DINO + projection head pipeline

**`dino_feature_extractor_tinyimagenet.ipynb`** (DINO, Tiny ImageNet):
- Extracts features for all Tiny ImageNet splits
- Larger dataset (~250 MB); extraction takes ~30–45 minutes
- Saves: `tinyimagenet_dino_train_embeddings.pt`, `tinyimagenet_dino_test_embeddings.pt`
- Use for: Extended evaluation on larger dataset

### Where to Place Downloaded Embeddings

After downloading from Colab/Kaggle, move embeddings to the correct directory:

```bash
# Expected structure
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

Pipelines will automatically detect cached embeddings and skip extraction.

### Troubleshooting Colab

**Issue**: `ModuleNotFoundError: No module named 'torch'`
- **Solution**: Colab comes with PyTorch pre-installed. Restart kernel if error persists.

**Issue**: Storage quota exceeded on Colab
- **Solution**: Download embeddings to Google Drive mid-extraction. Use `rm -rf /content/*` to clear temporary storage.

**Issue**: Google Drive mount fails
- **Solution**: Ensure the mount cell runs successfully. Re-run with authentication if needed.

**Issue**: Embeddings file is very large (~1GB+)
- **Solution**: This is normal for large datasets. Split downloads into parts or compress before transfer.

---

### DataManager

The `DataManager` class is a core utility that handles dataset and model management across all pipelines:

- **Dataset Loading**: Automatically downloads and caches datasets (CIFAR-10, STL-10, etc.) from torchvision
- **Embedding Storage**: Saves and loads cached embeddings to avoid redundant extraction
- **Model Registry**: Maintains a `.toml` registry file per dataset tracking all stored embeddings and models
- **Local Images**: Can load custom images from local folders with automatic preprocessing
- **Auto-Directory Creation**: Creates `embeddings/<dataset>/` and `models/<dataset>/` directories as needed

**Example Usage**:
```python
from dino_hdbscan.data_manager import DataManager
from torchvision import datasets

manager = DataManager(path='./data/stl10', batch_size=64, device='cuda')

# Auto-downloads STL-10 to ./data/stl10 on first run
train_loader = manager.get_loader(source=datasets.STL10, train=True)

# Caches embeddings
manager.store_embedding({'features': features, 'labels': labels}, name='my_features')

# Retrieves cached embeddings
features, labels = manager.load_embedding('my_features')
```

---

### Pipeline 1: Deep Embedded Clustering (DEC)

**Overview**: Combines ResNet-50 embeddings (2048-dim) with PCA reduction (50-dim) and Deep Embedded Clustering for unsupervised clustering on CIFAR-10.

**Dataset**: CIFAR-10 (auto-downloaded, ~160 MB)

**Training**
```bash
cd src/DEC-pipeline
python main.py
```

This will:
1. Load CIFAR-10 training set (auto-downloads if needed)
2. Extract ResNet-50 embeddings
3. Reduce to 50 dimensions via PCA
4. Train Deep Embedded Clustering model (10 epochs default)
5. Save model to `./models/cifar10/`
6. Print NMI, ARI, Silhouette metrics

**Testing**
```bash
python test_cluster.py
```

This evaluates the trained model on 1,000 unseen CIFAR-10 test images and generates cluster galleries.

**Configuration**: Edit hardcoded values in `main.py` (batch_size, epochs, n_clusters, embedding_dim).

---

### Pipeline 2: UMAP + HDBSCAN

**Overview**: Uses ResNet-50 embeddings (2048-dim) with UMAP reduction (2-dim) and HDBSCAN for density-based clustering on STL-10. Provides direct 2D visualization without additional dimensionality reduction.

**Dataset**: STL-10 (auto-downloaded, ~2.3 GB)

**Training**
```bash
cd src/hdbscan_pipeline
python main.py
```

This will:
1. Load STL-10 training set (auto-downloads if needed)
2. Extract ResNet-50 embeddings (2048-dim)
3. Reduce to 2-dim via UMAP (for direct visualization)
4. Cluster with HDBSCAN (min_cluster_size=50, min_samples=5)
5. Save HDBSCAN and UMAP models to `./models/stl10/`
6. Print NMI, ARI, Silhouette scores (computed on non-noise points only)
7. Show cluster statistics

**Testing**
```bash
python test_cluster.py
```

Evaluates clustering on 1,000 unseen STL-10 test images and generates per-cluster galleries.

**Configuration**: Edit constants in `main.py` (min_cluster_size, min_samples, umap_model_path).

---

### Pipeline 3: DINO + UMAP + HDBSCAN with Projection Head

**Overview**: Uses frozen DINO ViT-S/8 (384-dim) with UMAP reduction (50-dim), HDBSCAN pseudo-labeling, and a trainable MLP projection head for refined clustering on STL-10. Includes optional desktop UI for interactive clustering.

**Dataset**: STL-10 (auto-downloaded, ~2.3 GB)

**Training**
```bash
python -m dino_hdbscan.run_pipeline
```

This will:
1. Load STL-10 training set (auto-downloads if needed)
2. Extract frozen DINO ViT-S/8 features (384-dim, cached)
3. Reduce to 50-dim via UMAP
4. Generate HDBSCAN pseudo-labels with automatic noise reassignment
5. Mine k-nearest neighbors (k=10)
6. Train MLP projection head (384→256→128-dim) with supervised contrastive loss (50 epochs)
7. Save projection head and UMAP model to `./models/stl10/`

**Testing**
```bash
python -m dino_hdbscan.test_pipeline
```

Evaluates the trained projection head on 1,000 STL-10 test images and reports NMI, ARI, Silhouette scores.

**Custom Image Clustering**
```bash
# Edit src/dino_hdbscan/run_custom.py: set IMAGE_FOLDER to your folder path
python -m dino_hdbscan.run_custom
```

Clusters any folder of images using trained models and saves per-cluster galleries to `./results/galleries_custom/`.

**Desktop UI — Interactive Image Clustering**
```bash
python -m dino_hdbscan.main
```

Launches an interactive PySide6 Qt application from the project root for real-time clustering visualization and parameter tuning.

#### GUI Quick Start

**Step 1: Launch the Application**
```bash
python -m dino_hdbscan.main
```

This opens the **XCluster** desktop UI with:
- Left sidebar: folder selection, run controls, and advanced parameters
- Right panel: statistics, cluster list, and image gallery

**Step 2: Select Images to Cluster**
1. Click the **Browse** button in the sidebar
2. Navigate to your image folder (folder must contain `.jpg`, `.png`, `.jpeg`, or `.bmp` images)
3. **Recommended test folder**: `./data/stl10_test_input/` (pre-populated with 200+ diverse images)
4. Click **Select Folder** to confirm

**Step 3: Configure (Optional)**
- **Batch Size** (default: 16) — Reduce if you have limited GPU memory
- **Min Samples** (default: 2) — Lower values create more clusters; increase for fewer, larger clusters

**Step 4: Run Clustering**
1. Click the **Run Clustering** button
2. The UI will show progress: *"Extracting features..."* → *"Reducing dimensions..."* → *"Clustering..."*
3. When complete, the UI displays:
   - **Cluster statistics**: Total clusters found, noise images
   - **Cluster list**: Interactive chips showing each cluster (clickable)
   - **Image gallery**: Shows images from selected cluster with confidence scores

**Step 5: Explore Results**
- Click any cluster chip to view its images (sorted by confidence)
- Use the **Export Results** button to save cluster galleries as PNGs to `./results/galleries_custom/`

#### Test Images: Pre-Built Test Set

A **pre-built test set** is included in `./data/stl10_test_input/` with 200+ real-world images from diverse categories:
- **airplane** (18 images), **bird** (18), **bucket** (18), **car** (16), **cat** (15), **deer** (14), **dog** (18), **horse** (17), **monkey** (16), **ship** (13), **truck** (19)

**To use for GUI testing:**
1. Launch GUI: `python -m dino_hdbscan.main`
2. Click **Browse** → navigate to `./data/stl10_test_input`
3. Click **Run Clustering**
4. The pipeline should find ~10 clusters (one per object category)

#### Generating Custom Test Images with `test_images_downloader.py`

The `test_images_downloader.py` script downloads real-world images and populates `./data/stl10_test_input/` automatically. This is useful for:
- **Testing on different image categories** — Edit search terms to test non-STL10 concepts
- **Validating transfer learning** — Does the pipeline cluster objects it was never trained on?

**How to use:**
```bash
python src/dino_hdbscan/test_images_downloader.py
```

This downloads 20 images for each of 10 search terms from **LoremFlickr** (a reliable free image service) and saves them to `./data/stl10_test_input/`. The script is idempotent — re-running skips existing images.

**To test different image types:**
1. Open `src/dino_hdbscan/test_images_downloader.py`
2. Find this section:
   ```python
   search_terms = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]
   ```
3. Edit `search_terms` to any object categories you want to test (e.g., `["dog", "cat", "chair", "table", "laptop"]`)
4. Run the script:
   ```bash
   python src/dino_hdbscan/test_images_downloader.py
   ```
5. Images are saved with filenames: `IMG_<date>_<search_term>_<index>.jpg`
6. Open the GUI and test clustering on your new images to see if the pipeline can cluster objects beyond STL-10!

**Example Custom Test:**
```python
# Modify search_terms in test_images_downloader.py:
search_terms = ["laptop", "mouse", "keyboard", "monitor", "desk"]
# Then run the script and test in the GUI to see clustering on office equipment
```

**Configuration**: Edit constants in `src/dino_hdbscan/run_pipeline.py` (DEVICE, BATCH_SIZE, EPOCHS, LR, UMAP_DIMS, MIN_CLUSTER_SIZE, KNN_K).

---

## Required Datasets

The following datasets are required to train each pipeline. They are **automatically downloaded** on first run, but you can also download them manually:

| Pipeline | Dataset | Download Link | Size | First Run |
|----------|---------|---------------|------|-----------|
| DEC | CIFAR-10 | [Official](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) | ~160 MB | Auto-downloads to `data/cifar10/` |
| HDBSCAN | STL-10 | [Official](https://cs.stanford.edu/~acoates/stl10/) | ~2.3 GB | Auto-downloads to `data/stl10/` |
| DINO + Projection | STL-10 | [Official](https://cs.stanford.edu/~acoates/stl10/) | ~2.3 GB | Auto-downloads to `data/stl10/` |

**Manual Download** (optional):
1. Download the dataset from the link above
2. Extract to `./data/<dataset_name>/`
3. Run the pipeline — it will use existing files instead of re-downloading

**Cached Embeddings**:
Once extracted, DINO embeddings are cached to `./embeddings/stl10/` to avoid redundant computation on subsequent runs. Delete the cache folder to force re-extraction.

## Testing Each Pipeline

### Quick Test (Verify Installation)
```bash
# DEC Pipeline
cd src/DEC-pipeline && python -c "from feature_extractor import FeatureExtractor; print('✓ DEC pipeline imports OK')"

# HDBSCAN Pipeline
cd ../hdbscan_pipeline && python -c "from feature_extractor import FeatureExtractor; print('✓ HDBSCAN pipeline imports OK')"

# DINO Pipeline
cd ../dino_hdbscan && python -c "from feature_extractor import DinoFeatureExtractor; print('✓ DINO pipeline imports OK')"
```

### Full End-to-End Test
```bash
# 1. Train DEC
cd src/DEC-pipeline
python main.py
python test_cluster.py

# 2. Train HDBSCAN pipeline
cd ../hdbscan_pipeline
python main.py
python test_cluster.py

# 3. Train DINO pipeline (from project root)
cd ../..
python -m dino_hdbscan.run_pipeline
python -m dino_hdbscan.test_pipeline
```

### Expected Outputs

**Metrics** (displayed in terminal):
- **NMI** (Normalized Mutual Information): 0–1 range, higher is better
- **ARI** (Adjusted Rand Index): -1–1 range, higher is better (>0.3 is good)
- **Silhouette Score**: -1–1 range, higher indicates well-separated clusters

**Artifacts**:
- Trained models saved to `./models/*/`
- Embeddings cached to `./embeddings/*/`
- Results metrics saved to `./results/*.pt`
- Cluster galleries saved to `./results/galleries_*/`

## Configuration

### Global Config (`config.toml`)
```toml
[paths]
embeddings_dir = "embeddings"
models_dir = "models"
```

### Dataset Config (e.g., `data/stl10/stl10.toml`)
Each dataset directory gets a `.toml` registry file that tracks all embeddings and models created from that dataset. Automatically created upon first run.

### Pipeline-Specific Config
Edit hardcoded constants in pipeline entry points:
- `src/DEC-pipeline/main.py`: batch_size, epochs, n_clusters, embedding_dim
- `src/hdbscan_pipeline/main.py`: min_cluster_size, min_samples, umap dimensions
- `src/dino_hdbscan/run_pipeline.py`: DEVICE, BATCH_SIZE, EPOCHS, LR, UMAP_DIMS, MIN_CLUSTER_SIZE, KNN_K

## GPU/CPU Usage

- **Default**: Auto-detects GPU (CUDA 11.8+); falls back to CPU
- **Force CPU**: Manually set `DEVICE = 'cpu'` in pipeline scripts
- **VRAM requirements**: ~4GB for DINO extraction, ~2GB for training

## Common Issues

**Issue**: `ModuleNotFoundError: No module named 'dino_hdbscan'` when running from pipeline directory
- **Solution**: Run DINO pipeline from project root using module syntax:
  ```bash
  cd Image-clustering-FYP
  python -m dino_hdbscan.run_pipeline
  python -m dino_hdbscan.test_pipeline
  python -m dino_hdbscan.main      # For desktop UI
  ```

**Issue**: Slow embedding extraction on first run
- **Solution**: DINO embeddings are cached to `./embeddings/stl10/` and `./embeddings/cifar10/`. Subsequent runs reuse cached features. Delete cache folder to force re-extraction:
  ```bash
  rm -r embeddings/stl10/  # On Windows: rmdir /s embeddings\stl10
  ```

**Issue**: Out of Memory (OOM)
- **Solution**: Reduce `BATCH_SIZE` in pipeline scripts or limit dataset size with the `INFERENCE_LIMIT` parameter:
  ```python
  BATCH_SIZE = 32  # Instead of 64
  INFERENCE_LIMIT = 500  # Test on subset
  ```

**Issue**: HDBSCAN produces mostly noise (label -1)
- **Solution**: Lower `min_cluster_size` and `min_samples` parameters in pipeline scripts. Start with 10 and 2 for smaller datasets:
  ```python
  MIN_CLUSTER_SIZE = 10
  MIN_SAMPLES = 2
  ```

**Issue**: Datasets not downloading automatically
- **Solution**: Ensure write permissions in `./data/` directory. Manually download and extract to:
  - CIFAR-10 → `./data/cifar10/`
  - STL-10 → `./data/stl10/`

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) file for full details.

### MIT License Summary

You are free to:
- ✅ Use this project for personal, educational, and commercial purposes
- ✅ Modify and adapt the code
- ✅ Distribute copies (with proper attribution)

You must:
- 📋 Include the original copyright notice and license in any distributions
- 📋 Include a copy of the LICENSE file

This project is provided **AS-IS** without warranty. See full LICENSE file for complete terms.

---

## Citation

If you use this project in your research or work, please cite it as:

```bibtex
@misc{imageclusteringfyp2026,
  title={Image Clustering System: Unsupervised Deep Learning Pipeline},
  author={Uche},
  year={2026},
  howpublished={GitHub FYP Project},
  url={https://github.com/imnotuche/Image-clustering-FYP}
}
```

## References

- **DINO**: Caron et al., "Emerging Properties in Self-Supervised Vision Transformers" (ICCV 2021)
- **HDBSCAN**: McInnes et al., "Density-based clustering based on hierarchical density estimates" (2017)
- **UMAP**: McInnes et al., "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction" (2018)
- **DEC**: Xie et al., "Unsupervised Deep Embedding for Clustering Analysis" (ICML 2016)
- **Supervised Contrastive Learning**: Khosla et al., "Supervised Contrastive Learning" (NeurIPS 2020)

---

## Author

**Uche** — Final Year Project (FYP)

For questions or issues, please open an issue on GitHub.


