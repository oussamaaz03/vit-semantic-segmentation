#!/usr/bin/env python
"""
Calcul de la matrice de confusion complète (150×150)
pour le validation set ADE20K
"""

import torch
from pathlib import Path
from segm.data.factory import create_dataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

def compute_confusion_matrix(checkpoint_path, use_ema=True, num_images=2000):
    """
    Calcule la matrice de confusion complète 150×150
    """
    print("=" * 80)
    print("CALCUL DE LA MATRICE DE CONFUSION - ADE20K VALIDATION SET")
    print("=" * 80)
    
    # Charger checkpoint
    print(f"\n[1/6] Chargement du checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Charger modèle
    print("[2/6] Création du modèle...")
    if "model_cfg" in checkpoint:
        model_cfg = checkpoint["model_cfg"]
    else:
        # Configuration par défaut (ViT-Small)
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
    print(f"[INFO] Device: {device}")
    model.to(device)
    model.eval()
    
    # Créer dataset de validation
    print("[3/6] Chargement du dataset de validation ADE20K...")
    dataset = create_dataset({
        "dataset": "ade20k",
        "image_size": 512,
        "crop_size": 512,
        "split": "val"
    })
    
    val_loader = DataLoader(
        dataset,
        batch_size=4,  # Batch size plus petit pour économiser mémoire
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"[INFO] Nombre d'images de validation: {len(dataset)}")
    print(f"[INFO] Traitement de {num_images} images maximum")
    
    # Initialiser matrice de confusion 150 × 150
    print("[4/6] Calcul de la matrice de confusion...")
    confusion_matrix = np.zeros((150, 150), dtype=np.int64)
    
    num_processed = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Construction matrice"):
            if num_processed >= num_images:
                break
                
            images = batch["img"].to(device)
            targets = batch["gt_semantic_seg"].to(device)
            
            # Prédiction
            outputs = model(images)
            preds = outputs.argmax(1)
            
            # Accumuler dans matrice de confusion
            for pred, target in zip(preds, targets):
                pred_np = pred.cpu().numpy().flatten()
                target_np = target.cpu().numpy().flatten()
                
                # Pour chaque pixel, incrémenter confusion_matrix[true_class, pred_class]
                for true_label, pred_label in zip(target_np, pred_np):
                    confusion_matrix[true_label, pred_label] += 1
                
                num_processed += 1
                if num_processed >= num_images:
                    break
    
    print(f"\n[INFO] {num_processed} images traitées")
    print(f"[INFO] Total pixels comptés: {confusion_matrix.sum():,}")
    
    # Sauvegarder matrice brute
    print("\n[5/6] Sauvegarde de la matrice de confusion...")
    output_dir = Path("confusion_matrix_results")
    output_dir.mkdir(exist_ok=True)
    
    # Sauvegarder en .npy (format numpy)
    np.save(output_dir / "confusion_matrix_150x150.npy", confusion_matrix)
    print(f"✓ Matrice sauvegardée: {output_dir / 'confusion_matrix_150x150.npy'}")
    
    # Sauvegarder en .txt (lisible)
    np.savetxt(output_dir / "confusion_matrix_150x150.txt", confusion_matrix, fmt='%d')
    print(f"✓ Matrice sauvegardée (txt): {output_dir / 'confusion_matrix_150x150.txt'}")
    
    # Statistiques sur la matrice
    print("\n[INFO] Statistiques de la matrice de confusion:")
    diagonal = np.diag(confusion_matrix)
    total_per_class = confusion_matrix.sum(axis=1)
    
    # Accuracy par classe (recall)
    class_accuracy = np.zeros(150)
    for i in range(150):
        if total_per_class[i] > 0:
            class_accuracy[i] = diagonal[i] / total_per_class[i]
    
    print(f"  - Pixels total:           {confusion_matrix.sum():,}")
    print(f"  - Pixels correctement classifiés: {diagonal.sum():,}")
    print(f"  - Accuracy globale:       {(diagonal.sum() / confusion_matrix.sum() * 100):.2f}%")
    print(f"  - Recall moyen (classes): {(class_accuracy.mean() * 100):.2f}%")
    
    # Sauvegarder statistiques
    with open(output_dir / "confusion_matrix_stats.txt", "w") as f:
        f.write("STATISTIQUES MATRICE DE CONFUSION\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Images traitées:          {num_processed}\n")
        f.write(f"Pixels total:             {confusion_matrix.sum():,}\n")
        f.write(f"Pixels bien classifiés:   {diagonal.sum():,}\n")
        f.write(f"Accuracy globale:         {(diagonal.sum() / confusion_matrix.sum() * 100):.2f}%\n")
        f.write(f"Recall moyen par classe:  {(class_accuracy.mean() * 100):.2f}%\n\n")
        
        f.write("RECALL PAR CLASSE (top 20):\n")
        sorted_indices = np.argsort(class_accuracy)[::-1]
        for i in sorted_indices[:20]:
            if total_per_class[i] > 0:
                f.write(f"  Classe {i:3d}: {class_accuracy[i]*100:6.2f}% ({int(total_per_class[i]):,} pixels)\n")
    
    print(f"✓ Statistiques sauvegardées: {output_dir / 'confusion_matrix_stats.txt'}")
    
    # Visualisation
    print("\n[6/6] Génération de visualisations...")
    
    # Normaliser par ligne (pour voir la distribution des prédictions)
    confusion_normalized = np.zeros_like(confusion_matrix, dtype=float)
    for i in range(150):
        if total_per_class[i] > 0:
            confusion_normalized[i, :] = confusion_matrix[i, :] / total_per_class[i]
    
    # Plot 1: Heatmap complète (échelle log)
    plt.figure(figsize=(20, 18))
    # Utiliser échelle log pour mieux voir les petites valeurs
    confusion_log = np.log10(confusion_matrix + 1)  # +1 pour éviter log(0)
    sns.heatmap(confusion_log, cmap="YlOrRd", cbar_kws={'label': 'log10(count + 1)'})
    plt.title("Matrice de Confusion 150×150 (échelle log)", fontsize=16, weight='bold')
    plt.xlabel("Classe Prédite", fontsize=14)
    plt.ylabel("Classe Réelle (Ground Truth)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix_full_log.png", dpi=150)
    print(f"✓ Heatmap (log) sauvegardée: {output_dir / 'confusion_matrix_full_log.png'}")
    plt.close()
    
    # Plot 2: Heatmap normalisée (par ligne = recall)
    plt.figure(figsize=(20, 18))
    sns.heatmap(confusion_normalized, cmap="RdYlGn", vmin=0, vmax=1, 
                cbar_kws={'label': 'Proportion'})
    plt.title("Matrice de Confusion Normalisée (Recall par classe)", fontsize=16, weight='bold')
    plt.xlabel("Classe Prédite", fontsize=14)
    plt.ylabel("Classe Réelle (Ground Truth)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix_normalized.png", dpi=150)
    print(f"✓ Heatmap normalisée sauvegardée: {output_dir / 'confusion_matrix_normalized.png'}")
    plt.close()
    
    # Plot 3: Diagonale (accuracy par classe)
    plt.figure(figsize=(24, 6))
    x = np.arange(150)
    plt.bar(x, class_accuracy * 100, color='steelblue', alpha=0.7)
    plt.axhline(y=class_accuracy.mean() * 100, color='red', linestyle='--', 
                label=f'Moyenne: {class_accuracy.mean()*100:.1f}%')
    plt.xlabel("Classe ID", fontsize=12)
    plt.ylabel("Recall (%)", fontsize=12)
    plt.title("Recall (Accuracy) par Classe - 150 classes ADE20K", fontsize=14, weight='bold')
    plt.xticks(np.arange(0, 150, 10))
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "recall_per_class.png", dpi=150)
    print(f"✓ Graphique recall sauvegardé: {output_dir / 'recall_per_class.png'}")
    plt.close()
    
    print("\n" + "=" * 80)
    print("✅ TERMINÉ!")
    print("=" * 80)
    print(f"\n📁 Résultats dans: {output_dir}/")
    print(f"   - confusion_matrix_150x150.npy (matrice numpy)")
    print(f"   - confusion_matrix_150x150.txt (matrice texte)")
    print(f"   - confusion_matrix_stats.txt (statistiques)")
    print(f"   - confusion_matrix_full_log.png (heatmap complète)")
    print(f"   - confusion_matrix_normalized.png (heatmap normalisée)")
    print(f"   - recall_per_class.png (graphique recall)")
    
    return confusion_matrix, confusion_normalized

if __name__ == "__main__":
    checkpoint_path = "logs/vit_small_cloud/checkpoint.pth"
    compute_confusion_matrix(checkpoint_path, use_ema=True, num_images=2000)
