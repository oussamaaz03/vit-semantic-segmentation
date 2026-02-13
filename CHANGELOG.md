# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-02-13

### 🎉 Initial Release

#### Added
- **Vision Transformer Small (ViT-S/16)** architecture for semantic segmentation
- **Mask Transformer** decoder (2 layers)
- **Hybrid Loss Function** (Cross-Entropy + Dice Loss, α=0.5)
- **AdamW Optimizer** with weight decay (5e-4)
- **Polynomial Learning Rate Scheduler** with warmup (5 epochs)
- **Exponential Moving Average (EMA)** for stable training
- **Label Smoothing** (ε=0.1) for better generalization
- **Gradient Clipping** for training stability
- **Data Augmentation**: RandomScale, RandomCrop, RandomFlip, ColorJitter
- Complete training pipeline on ADE20K dataset (150 classes)
- Comprehensive evaluation scripts
- Visualization tools (per-class IoU, heatmaps, scatter plots)
- Professional README with detailed methodology
- Pipeline architecture diagram
- Example segmentations on diverse scenes

#### Performance
- **Mean IoU (mIoU):** 43.08% on ADE20K validation set (2000 images)
- **Pixel Accuracy:** 80.79%
- **Improvement over baseline:** +13.4% (38.00% → 43.08%)
- **Model size:** 27M parameters

#### Scripts
- `demo.py` - Quick demonstration script
- `Scripts/predict.py` - Single image inference
- `Scripts/evaluate_quantitative.py` - Complete evaluation on validation set
- `Scripts/visualize_results.py` - Generate analysis plots
- `Scripts/compute_confusion_matrix.py` - Confusion matrix computation

#### Documentation
- Comprehensive README.md with:
  - Installation guide
  - Quick start tutorial
  - Training instructions
  - Evaluation guide
  - Methodology explanation
  - Results visualization
  - BibTeX citation
- RAPPORT_PFE_FINAL.md - Complete project report
- STATUS_PROJET.md - Project status and checklist

#### Results & Visualizations
- `results/samples/` - Example segmentations
- `results/visualisations/` - Analysis plots:
  - Per-class IoU bar chart
  - IoU heatmap (150 classes)
  - IoU vs. frequency scatter plot
- `Pipeline_Segmenter_Ameliore.png` - Architecture diagram

#### Infrastructure
- MIT License
- Complete .gitignore for Python projects
- requirements.txt with dependencies
- setup.py for package installation

---

## [Unreleased]

### Planned Features
- [ ] Scientific paper (LaTeX)
- [ ] ArXiv submission
- [ ] Additional ablation studies
- [ ] Multi-scale evaluation
- [ ] Test-Time Augmentation optimization
- [ ] Confusion matrix analysis
- [ ] Attention map visualization
- [ ] Jupyter notebook tutorials
- [ ] Docker containerization
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Model quantization for deployment
- [ ] ONNX export
- [ ] Web demo (Gradio/Streamlit)

---

## Notes

### Training Details
- **Dataset:** ADE20K (20,210 training images, 2,000 validation)
- **Resolution:** 512×512 pixels
- **Batch Size:** 8
- **Epochs:** 48
- **Hardware:** NVIDIA A100 GPU (40GB VRAM)
- **Training Time:** ~48 hours

### Key Contributions
1. **Hybrid Loss** addresses class imbalance and improves small object detection
2. **AdamW + Polynomial LR** provides better convergence than SGD
3. **EMA** stabilizes predictions and improves generalization
4. **Label Smoothing** prevents overconfidence
5. **Complementary techniques** work synergistically (not additively)

### Known Limitations
- Performance on rare classes (<25th percentile) remains challenging (18.81% mIoU)
- High inter-image variance (σ=17.07%) indicates scene-dependent performance
- Some classes achieve 0% IoU due to extreme rarity (barrel, tray, shower, step)

---

## Links

- **Repository:** https://github.com/YOUR_USERNAME/vit-semantic-segmentation
- **Report:** See RAPPORT_PFE_FINAL.md
- **Issues:** https://github.com/YOUR_USERNAME/vit-semantic-segmentation/issues
- **Paper:** Coming soon

---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{vit-semantic-segmentation-2026,
  author = {Your Name},
  title = {Vision Transformer for Semantic Segmentation with Hybrid Loss Optimization},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/YOUR_USERNAME/vit-semantic-segmentation}}
}
```
