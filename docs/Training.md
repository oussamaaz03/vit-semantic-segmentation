# 🚀 Training Guide

Guide for training **Vision Transformer models for Semantic Segmentation**.

---

## ✅ Prerequisites

- ✅ Completed [Installation.md](Installation.md)
- ✅ ADE20K dataset downloaded (20,210 training images)
- ✅ NVIDIA GPU with 11GB+ VRAM
- ✅ 50GB+ free disk space

---

## ⚡ Quick Start

### Single GPU Training

```bash
python segm/train.py \
    --dataset ade20k \
    --backbone vit_small_patch16_384 \
    --decoder mask_transformer \
    --epochs 48 \
    --batch-size 8 \
    --lr 0.0001
```

**Training time:** ~24 hours on A100 (40GB)  
**Expected results:** mIoU ~43%, Pixel Accuracy ~81%

### Multi-GPU Training (4 GPUs)

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    segm/train.py \
    --dataset ade20k \
    --backbone vit_small_patch16_384 \
    --batch-size 8
```

**Training time:** ~6-8 hours  
**Effective batch size:** 4 × 8 = 32

---

## ⚙️ Configuration

### Main Config: `segm/config.yml`

```yaml
# Model
model_name: vit_small_patch16_384
d_model: 384
n_heads: 6
n_layers: 12

# Decoder
decoder:
  name: mask_transformer
  n_layers: 2
  dropout: 0.1

# Training
epochs: 48
batch_size: 8
base_lr: 0.0001
optimizer: adamw
weight_decay: 0.01

# Loss
loss:
  name: hybrid
  ce_weight: 0.5
  dice_weight: 0.5
  label_smoothing: 0.1

# Augmentation
augmentation:
  random_resize: [0.5, 2.0]
  random_crop: [512, 512]
  random_flip: true
  color_jitter: 0.4
```

---

## 🎯 Training Modes

### Resume Training

```bash
python segm/train.py \
    --resume logs/vit_small/checkpoint.pth \
    --epochs 60
```

### Fine-tuning

```bash
python segm/train.py \
    --pretrained checkpoint.pth \
    --epochs 20 \
    --lr 0.00001 \
    --freeze-encoder true
```

## 📊 Monitoring


**Metrics logged:**
- Loss curves (CE, Dice, Total)
- Learning rate
- mIoU per epoch
- Pixel accuracy
- Sample predictions

### Console Output

```
Epoch: [1/48]
Step [100/2527] Loss: 1.234 | LR: 0.0001 | ETA: 2h15m
  ├─ CE Loss: 0.876
  ├─ Dice Loss: 0.358
  ├─ mIoU: 28.45%
  └─ Pixel Acc: 72.34%

Validation [Epoch 1]:
  ├─ mIoU: 31.23%
  └─ Best mIoU: 31.23% ⭐
```

---

## 🔬 Advanced Options

### Mixed Precision (Faster + Less Memory)

```bash
python segm/train.py --amp true --batch-size 16
```

**Benefits:** 1.5-2× faster, 30-40% less memory

### Custom Loss Weights

```yaml
loss:
  ce_weight: 0.7    # Emphasize classification
  dice_weight: 0.3
```

### Learning Rate Scheduling

```yaml
scheduler:
  name: polynomial  # Default
  # OR
  name: cosine
  warmup_epochs: 5
```

### Data Augmentation

```yaml
augmentation:
  random_resize: [0.5, 2.0]
  random_crop: [512, 512]
  random_flip: true
  color_jitter: 0.4
  rand_aug: true
```

**Impact:** +3% mIoU improvement

---

### Out of Memory

```bash
# Option 1: Reduce batch size
--batch-size 4

# Option 2: Gradient accumulation
--batch-size 2 --accumulate-grad-batches 4

# Option 3: Mixed precision
--amp true
```

### Slow Training

```yaml
# Use more DataLoader workers
num_workers: 8
pin_memory: true
```

### Poor Validation Performance

```yaml
# Increase regularization
weight_decay: 0.05
label_smoothing: 0.15
dropout: 0.2

# More augmentation
rand_aug: true
```

---

## 📈 Training Tips

1. **Start small:** Test with 2 epochs first
2. **Monitor early:** Check TensorBoard after epoch 1
3. **Save frequently:** Checkpoint every 5 epochs
4. **Expected mIoU:**
   - Epoch 1: ~20-25%
   - Epoch 10: ~35-38%
   - Epoch 48: ~42-44%

---

## 🎯 Key Hyperparameters

| Parameter | Value | Impact |
|-----------|-------|--------|
| Learning rate | 0.0001 | High |
| Batch size | 8 | Medium |
| Weight decay | 0.01 | Medium |
| Dropout | 0.1 | Low-Medium |
| Label smoothing | 0.1 | Low |

---

## 📚 Next Steps

- **Evaluate:** See [Evaluation.md](Evaluation.md)
- **Deploy:** Export to ONNX for production
- **Experiment:** Try different backbones (ViT-Base, ViT-Large)

---

**Last Updated:** February 13, 2026
