# 📚 Documentation Complete

Bienvenue dans la documentation du projet **Vision Transformer for Semantic Segmentation**!

---

## 📖 Table des Matières

### 📌 Guides Disponibles

1. **[Installation](Installation.md)** ✅ - Guide complet d'installation
   - System requirements
   - Quick install (5 min)
   - Detailed installation
   - Dataset download
   - Troubleshooting

2. **[Training Guide](Training.md)** ✅ - Guide d'entraînement avancé
   - Configuration files
   - Single/Multi-GPU training
   - Hyperparameter tuning
   - Custom datasets
   - Advanced techniques (EMA, AMP, Custom losses)

3. **[Evaluation Guide](Evaluation.md)** ✅ - Évaluation et métriques
   - mIoU, Pixel Accuracy explained
   - Evaluation modes
   - Visualization tools
   - Error analysis
   - Performance benchmarking

### 🚧 Coming Soon

4. [API Reference](API.md) - Documentation de l'API (coming soon)
5. [FAQ](FAQ.md) - Questions fréquentes (coming soon)
6. [Deployment Guide](Deployment.md) - ONNX/TensorRT export (coming soon)

---

## 🚀 Démarrage Rapide

### Installation Express (5 minutes)

```bash
# 1. Clone le repository
git clone https://github.com/YOUR_USERNAME/vit-semantic-segmentation.git
cd vit-semantic-segmentation

# 2. Créer environnement
conda create -n segmenter python=3.8
conda activate segmenter

# 3. Installer dépendances
pip install -r requirements.txt

# 4. Télécharger checkpoint
# [Lien Google Drive ou commande]

# 5. Tester
python demo.py
```

### Premier Test (1 minute)

```bash
# Segmentation d'une image
python Scripts/predict.py --image path/to/image.jpg --checkpoint checkpoint.pth
```

---

## 📊 Résultats Principaux

| Métrique | Valeur |
|----------|--------|
| **Mean IoU** | 43.08% |
| **Pixel Accuracy** | 80.79% |
| **Paramètres** | 27M |
| **Dataset** | ADE20K (150 classes) |
| **Amélioration** | +13.4% vs baseline |

---

## 📘 Guides de Documentation

### ✅ Guides Disponibles

| Guide | Contenu | Status |
|-------|---------|--------|
| **[Installation.md](Installation.md)** | Installation complète, dataset, troubleshooting | ✅ Complete |
| **[Training.md](Training.md)** | Single/Multi-GPU, configuration, monitoring | ✅ Complete |
| **[Evaluation.md](Evaluation.md)** | Métriques (mIoU, PA), visualisation, benchmarking | ✅ Complete |

### 📌 Contenu de chaque guide:

**Installation.md** (~150 lignes):
- System requirements
- Quick install (5 min)
- Step-by-step guide
- Dataset download
- Troubleshooting (CUDA, mmcv, dataset)
- Verification tests

**Training.md** (~200 lignes):
- Quick start (single/multi-GPU)
- Configuration YAML
- Training modes (resume, fine-tuning)
- Monitoring (TensorBoard)
- Advanced options (Mixed Precision, augmentation)
- Troubleshooting (NaN, OOM)

**Evaluation.md** (~180 lignes):
- Metrics explained (mIoU, PA)
- Quick evaluation
- Visualization tools
- Performance summary
- Troubleshooting

---

## 🔬 Architecture

```
Input Image (512×512)
    ↓
Vision Transformer Small (ViT-S/16)
    ├── Patch Embedding (16×16 patches)
    ├── 12 Transformer Layers
    │   ├── d_model = 384
    │   ├── n_heads = 6
    │   └── MLP ratio = 4.0
    └── Output: [B, 1024, 384]
    ↓
Mask Transformer Decoder (2 layers)
    ├── Cross-Attention with image features
    ├── 150 class embeddings
    └── Patch-wise classification
    ↓
Segmentation Map (512×512, 150 classes)
```

---

## 🎯 Contributions Clés

### 1. Hybrid Loss Function
```python
Loss = 0.5 × CrossEntropy + 0.5 × DiceLoss
```
- **Pourquoi?** Cross-Entropy seule ignore les petits objets
- **Impact:** +5.08 mIoU (38% → 43.08%)

### 2. AdamW Optimizer
```python
AdamW(lr=1e-4, weight_decay=5e-4)
```
- **Pourquoi?** Meilleure généralisation que SGD sur ViT
- **Impact:** Convergence stable et rapide

### 3. Polynomial LR + Warmup
```python
Warmup: 0 → 1e-4 (5 epochs)
Decay: lr(t) = lr_max × (1 - t/T)^0.9
```
- **Pourquoi?** Évite divergence au début
- **Impact:** +1-2 mIoU final

### 4-6. EMA + Label Smoothing + Gradient Clipping
- **EMA:** Stabilise prédictions
- **Label Smoothing (ε=0.1):** Évite surconfiance
- **Gradient Clipping:** Évite explosions de gradients

---

## 📁 Structure du Projet

```
vit-semantic-segmentation/
├── README.md              # Documentation principale
├── demo.py                # Demo rapide
├── CHANGELOG.md           # Historique versions
├── checkpoint.pth         # Modèle entraîné
│
├── Scripts/               # Scripts utilisables
│   ├── predict.py         # Inference
│   ├── evaluate_quantitative.py
│   ├── visualize_results.py
│   └── compute_confusion_matrix.py
│
├── segm/                  # Package principal
│   ├── model/             # Modèles (ViT, Decoder)
│   ├── data/              # Datasets
│   ├── optim/             # Optimizers
│   ├── eval/              # Métriques
│   └── utils/             # Utilitaires
│
├── results/               # Résultats
│   ├── samples/           # Images segmentées
│   └── visualisations/    # Graphiques
│
└── docs/                  # Documentation (ici!)
```

---

## 🔗 Liens Utiles

- **GitHub:** [Repository](https://github.com/YOUR_USERNAME/vit-semantic-segmentation)
- **Paper:** Coming soon
- **Issues:** [Report a bug](https://github.com/YOUR_USERNAME/vit-semantic-segmentation/issues)
- **Dataset ADE20K:** [MIT CSAIL](http://sceneparsing.csail.mit.edu/)

---

## 📧 Support

- **Issues GitHub:** Pour bugs et features
- **Email:** your.email@university.edu
- **Discussion:** [GitHub Discussions](https://github.com/YOUR_USERNAME/vit-semantic-segmentation/discussions)

---

## 📄 License

Ce projet est sous license **MIT**. Voir [LICENSE](../LICENSE) pour détails.

---

**Dernière mise à jour:** 13 Février 2026
