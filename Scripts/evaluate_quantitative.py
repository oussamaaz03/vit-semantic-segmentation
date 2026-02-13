#!/usr/bin/env python
"""
Évaluation quantitative complète sur le validation set ADE20K officiel
- mIoU global
- mIoU par classe
- Analyse des classes rares
- Variance inter-images
"""

import sys
import torch
from pathlib import Path
from segm.eval.miou import MeanIntersectionOverUnion
from segm.model.factory import load_model
import yaml
from segm.data.factory import create_dataset
from tqdm import tqdm
import numpy as np

def evaluate_on_ade20k(checkpoint_path, use_ema=True):
    """
    Évaluation quantitative complète
    """
    print("=" * 80)
    print("ÉVALUATION QUANTITATIVE SUR ADE20K VALIDATION SET")
    print("=" * 80)
    
    # Charger checkpoint
    print(f"\n[1/5] Chargement du checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Charger modèle
    print("[2/5] Création du modèle...")
    if "model_cfg" in checkpoint:
        model_cfg = checkpoint["model_cfg"]
    else:
        # Configuration par défaut
        model_cfg = {
            "backbone": "vit_small_patch16_384",
            "image_size": [512, 512],
            "d_model": 384,
            "n_heads": 6,
            "n_layers": 12,
            "decoder": {"name": "mask_transformer", "n_layers": 2},
            "n_cls": 150
        }
    
    from segm.model.factory import create_segmenter
    model = create_segmenter(model_cfg)
    
    # Charger weights (EMA si disponible)
    if use_ema and "ema_model" in checkpoint:
        print("[INFO] Utilisation du modèle EMA")
        model.load_state_dict(checkpoint["ema_model"])
    else:
        print("[INFO] Utilisation du modèle standard")
        model.load_state_dict(checkpoint["model"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Créer dataset de validation
    print("[3/5] Chargement du dataset de validation ADE20K...")
    dataset = create_dataset({
        "dataset": "ade20k",
        "image_size": 512,
        "crop_size": 512,
        "split": "val"
    })
    
    from torch.utils.data import DataLoader
    val_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"[INFO] Nombre d'images de validation: {len(dataset)}")
    
    # Métriques
    print("[4/5] Calcul des métriques...")
    miou_metric = MeanIntersectionOverUnion(num_classes=150, ignore_index=0)
    
    # Statistiques par classe
    class_ious = []
    class_counts = np.zeros(150)
    image_ious = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Évaluation"):
            images = batch["img"].to(device)
            targets = batch["gt_semantic_seg"].to(device)
            
            # Prédiction
            outputs = model(images)
            preds = outputs.argmax(1)
            
            # Update metrics
            miou_metric.update(preds, targets)
            
            # Calcul IoU par image
            for pred, target in zip(preds, targets):
                pred_np = pred.cpu().numpy()
                target_np = target.cpu().numpy()
                
                # IoU pour cette image
                intersection = np.logical_and(pred_np == target_np, target_np != 0)
                union = np.logical_or(pred_np != 0, target_np != 0)
                if union.sum() > 0:
                    iou = intersection.sum() / union.sum()
                    image_ious.append(iou)
                
                # Compter classes présentes
                unique_classes = np.unique(target_np)
                for cls in unique_classes:
                    if cls != 0:  # Ignorer background
                        class_counts[cls] += 1
    
    # Résultats
    print("\n" + "=" * 80)
    print("[5/5] RÉSULTATS")
    print("=" * 80)
    
    results = miou_metric.compute()
    mean_iou = results["mean_iou"]
    pixel_acc = results.get("pixel_accuracy", 0)
    mean_acc = results.get("mean_accuracy", 0)
    
    print(f"\n📊 MÉTRIQUES GLOBALES:")
    print(f"  - Mean IoU:        {mean_iou:.2f}%")
    print(f"  - Pixel Accuracy:  {pixel_acc:.2f}%")
    print(f"  - Mean Accuracy:   {mean_acc:.2f}%")
    
    # Variance inter-images
    image_ious_arr = np.array(image_ious)
    print(f"\n📈 VARIANCE INTER-IMAGES:")
    print(f"  - IoU moyen:       {image_ious_arr.mean():.2f}%")
    print(f"  - Écart-type:      {image_ious_arr.std():.2f}%")
    print(f"  - IoU min:         {image_ious_arr.min():.2f}%")
    print(f"  - IoU max:         {image_ious_arr.max():.2f}%")
    print(f"  - IoU médian:      {np.median(image_ious_arr):.2f}%")
    
    # Classes rares (< 100 occurrences)
    rare_classes = np.where(class_counts < 100)[0]
    common_classes = np.where(class_counts >= 100)[0]
    
    print(f"\n🔍 ANALYSE PAR FRÉQUENCE DE CLASSES:")
    print(f"  - Classes rares (< 100 images):    {len(rare_classes)}")
    print(f"  - Classes communes (≥ 100 images): {len(common_classes)}")
    
    # Top 10 et Bottom 10 classes
    if "per_class_iou" in results:
        class_ious = results["per_class_iou"]
        sorted_idx = np.argsort(class_ious)
        
        print(f"\n🏆 TOP 10 CLASSES (meilleure IoU):")
        for i, idx in enumerate(sorted_idx[-10:][::-1], 1):
            print(f"  {i}. Classe {idx}: {class_ious[idx]:.2f}% ({int(class_counts[idx])} images)")
        
        print(f"\n⚠️  BOTTOM 10 CLASSES (pire IoU):")
        for i, idx in enumerate(sorted_idx[:10], 1):
            if class_counts[idx] > 0:
                print(f"  {i}. Classe {idx}: {class_ious[idx]:.2f}% ({int(class_counts[idx])} images)")
    
    # Sauvegarder résultats
    results_path = Path("evaluation_results.txt")
    with open(results_path, "w") as f:
        f.write("ÉVALUATION QUANTITATIVE - ADE20K VALIDATION SET\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Mean IoU: {mean_iou:.2f}%\n")
        f.write(f"Pixel Accuracy: {pixel_acc:.2f}%\n")
        f.write(f"Mean Accuracy: {mean_acc:.2f}%\n")
        f.write(f"\nNombre d'images évaluées: {len(dataset)}\n")
        f.write(f"Variance IoU (std): {image_ious_arr.std():.2f}%\n")
    
    print(f"\n✓ Résultats sauvegardés: {results_path}")
    print("=" * 80)
    
    return results

if __name__ == "__main__":
    checkpoint_path = "logs/vit_small_cloud/checkpoint.pth"
    evaluate_on_ade20k(checkpoint_path, use_ema=True)
