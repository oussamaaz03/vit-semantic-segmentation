# 📊 Evaluation Guide

Guide for evaluating **Vision Transformer Semantic Segmentation** models.

---

## 🎯 Available Metrics

| Metric | Description | Our Score |
|--------|-------------|-----------|
| **mIoU** | Mean Intersection over Union | 43.08% |
| **Pixel Accuracy** | Correct pixels / Total pixels | 80.79% |
| **Class IoU** | IoU per semantic class | Varies |
| **Inference Time** | Speed (ms/image) | ~180ms |

---

## ⚡ Quick Evaluation

### Single Image

```bash
# Quick demo
python demo.py

# Custom image
python demo.py path/to/image.jpg

# With GPU
python demo.py image.jpg --device cuda
```

**Output:** `*_demo_segmentation.png`, `*_demo_overlay.png`

### Full Validation (2,000 images)

```bash
python Scripts/evaluate_quantitative.py
```

**Expected output:**
```
Images évaluées: 2000
mIoU moyen: 43.08%
Pixel Accuracy: 80.79%
Temps total: 6m 12s
```

**Files generated:**
- `evaluation_results.txt` - Summary
- `per_class_iou.csv` - IoU for 150 classes
- `confusion_matrix.npy` - 150×150 matrix

### Visualizations

```bash
python Scripts/visualize_results.py
```

**Outputs:**
- `top_10_classes_bar.png` - Best classes
- `bottom_10_classes_bar.png` - Worst classes
- `confusion_matrix_heatmap.png` - Full heatmap
- `predictions_grid.png` - Sample predictions

---

## 📐 Metrics Explained

### mIoU (Mean Intersection over Union)

**Formula:**
```
IoU = Intersection / Union
    = TP / (TP + FP + FN)

mIoU = Average IoU across all 150 classes
```

**Example:** Class "car"
- Prediction: 1000 pixels
- Ground truth: 1200 pixels
- Overlap: 800 pixels
- IoU = 800 / (1000 + 1200 - 800) = 57.14%

**Interpretation:**
- 0-20%: Poor
- 20-40%: Fair
- 40-60%: Good ⭐ (Our level)
- 60-80%: Very Good
- 80-100%: Excellent

### Pixel Accuracy

**Formula:**
```
PA = Correct Pixels / Total Pixels
```

**Example:**
- Total: 262,144 pixels (512×512)
- Correct: 211,799 pixels
- PA = 80.79%

**Why PA ≠ mIoU:**
- PA dominated by common classes (sky, road)
- mIoU penalizes rare class failures
- Our PA (80.79%) > mIoU (43.08%)

### Per-Class Performance

**Top 5 Classes:**

| Class | IoU | Why Good |
|-------|-----|----------|
| Tent | 91.67% | Large, distinct |
| Sky | 83.35% | Homogeneous |
| Building | 77.24% | Clear boundaries |
| Road | 76.89% | Uniform texture |
| Tree | 72.45% | Recognizable shape |

**Bottom 5 Classes:**

| Class | IoU | Issue |
|-------|-----|-------|
| Fan | 5.23% | Small, complex |
| Pot | 2.14% | Variable appearance |
| Barrel | 0.00% | Too rare |
| Tray | 0.00% | Never detected |

---

## 🔍 Evaluation Modes

### Fast Evaluation (100 images)

```bash
python Scripts/evaluate_quantitative.py --num-images 100
# ~30 seconds, ±1% accuracy
```


### Speed Benchmark

```bash
python Scripts/benchmark_speed.py
```

**Results:**
```
Device: NVIDIA A100
Single image: 180ms (5.5 FPS)
Batch (8 images): 68ms/image (14.7 FPS)
```

---

## 🎨 Visualization

### Prediction Overlays

```bash
python Scripts/visualize_predictions.py --input image.jpg
```

**Generates:**
- `image_segmentation.png` - Color mask
- `image_overlay.png` - Overlay
- `image_comparison.png` - Side-by-side

## 🐛 Troubleshooting

### Slow Evaluation

```bash
# Use batch inference
python evaluate.py --batch-size 8

# Reduce resolution
python evaluate.py --eval-size 384
```

### Low mIoU but High PA

**Issue:** Model ignores rare classes

**Solutions:**
```python
# Class-balanced loss
criterion = FocalLoss(alpha=class_weights)

# Oversample rare classes
sampler = ClassBalancedSampler(dataset)
```

### Different Results Each Run

**Issue:** Non-deterministic operations

**Solution:**
```python
model.eval()
torch.backends.cudnn.deterministic = True
torch.manual_seed(42)
```

--

## 📊 Performance Summary

### Our Results (ViT Small)

```
Dataset: ADE20K (2,000 validation images)
Model: ViT-S/16 (27M params)

Overall:
├─ mIoU: 43.08%
├─ Pixel Accuracy: 80.79%
├─ Inference Speed: 180ms/image (A100)
└─ Training Time: 24h (single A100)

Top Classes (>70% IoU):
├─ Tent (91.67%)
├─ Sky (83.35%)
├─ Building (77.24%)
├─ Road (76.89%)
└─ Tree (72.45%)

Challenges:
├─ Rare classes (6 classes at 0%)
├─ Small objects (<50px)
└─ Class confusion (sky↔ceiling)
```

---

## 📈 Comparison

| Method | mIoU | Params | Speed |
|--------|------|--------|-------|
| FCN-8s | 29.4% | 134M | 45ms |
| DeepLab v3+ | 37.8% | 41M | 120ms |
| **Ours (ViT-S)** | **43.1%** | **27M** | **180ms** |
| Segformer | 45.6% | 84M | 220ms |

---
## 📚 Next Steps

- **Improve:** See [Training.md](Training.md) for advanced techniques
- **Deploy:** Export to ONNX/TensorRT
- **Analyze:** Error analysis for improvements

---

**Last Updated:** February 13, 2026
