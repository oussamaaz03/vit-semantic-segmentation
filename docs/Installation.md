# 📦 Installation Guide

Complete installation guide for **Vision Transformer Semantic Segmentation**.

---

## 💻 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.8+ | 3.8 - 3.10 |
| **RAM** | 16 GB | 32 GB+ |
| **GPU** | NVIDIA GTX 1080 Ti (11GB) | NVIDIA A100 (40GB) |
| **CUDA** | 11.3+ | 11.8+ |
| **Storage** | 50 GB | 100 GB+ |

---

## ⚡ Quick Installation

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/vit-semantic-segmentation.git
cd vit-semantic-segmentation

# 2. Create environment
conda create -n segmenter python=3.8 -y
conda activate segmenter

# 3. Install PyTorch (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download dataset
python -m segm.scripts.prepare_ade20k

# 6. Test installation
python demo.py
```

---

## 🔧 Step-by-Step Installation

### 1. Python Environment

```bash
# Using Conda (recommended)
conda create -n segmenter python=3.8 -y
conda activate segmenter

# Verify
python --version  # Should show Python 3.8.x
```

### 2. Install PyTorch

```bash
# CUDA 11.8 (recommended)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# CPU only (no GPU)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu
```

**Verify CUDA:**
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key packages:** `timm==0.4.12`, `mmcv==1.3.8`, `mmsegmentation==0.14.1`, `einops`, `matplotlib`

### 4. Download Dataset

```bash
# Automatic download (recommended)
python -m segm.scripts.prepare_ade20k

# Expected structure:
# data/ADEChallengeData2016/
#   ├── images/training/    (20,210 images)
#   ├── images/validation/  (2,000 images)
#   ├── annotations/training/
#   └── annotations/validation/
```

### 5. Download Checkpoint

```bash
# From GitHub Releases
wget https://github.com/YOUR_USERNAME/vit-semantic-segmentation/releases/download/v1.0/checkpoint.pth

# OR from Google Drive
pip install gdown
gdown https://drive.google.com/uc?id=FILE_ID -O checkpoint.pth
```

---

## 🐛 Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python train.py --batch-size 4  # Instead of 8
```

### mmcv Build Failure

```bash
# Install prebuilt wheel
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
```

### Dataset Not Found

```bash
# Re-download
python -m segm.scripts.prepare_ade20k

# Or set explicit path
export DATASET=/path/to/ade20k
```

---

## ✅ Verification

```bash
# Test 1: Imports
python -c "import torch; import segm; print('✅ All imports successful')"

# Test 2: CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Test 3: Demo
python demo.py
```

**Expected output:**
```
✅ Checkpoint: checkpoint.pth
✅ Image: results\samples\Figure_2.png
[1/4] Chargement du checkpoint... ✓
[2/4] Création du modèle... ✓
[3/4] Prétraitement de l'image... ✓
[4/4] Segmentation en cours... ✓
✅ Segmentation terminée!
```

---

## 📚 Next Steps

- **Training:** See [Training.md](Training.md)
- **Evaluation:** See [Evaluation.md](Evaluation.md)
- **Issues:** [Report a bug](https://github.com/YOUR_USERNAME/vit-semantic-segmentation/issues)

---

**Last Updated:** February 13, 2026
