#!/usr/bin/env python3
"""
DEMO RAPIDE - Segmentation Sémantique avec Vision Transformer
Utilisation: python demo.py [chemin_image]
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pathlib import Path

# Importer depuis segm
from segm.model.factory import create_segmenter

# ADE20K Palette (150 classes)
ADE20K_PALETTE = [
    [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3],
    [120, 120, 80], [140, 140, 140], [204, 5, 255], [230, 230, 230], [4, 250, 7],
    [224, 5, 255], [235, 255, 7], [150, 5, 61], [120, 120, 70], [8, 255, 51],
    [255, 6, 82], [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
    [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255], [255, 7, 71],
    [255, 9, 224], [9, 7, 230], [220, 220, 220], [255, 9, 92], [112, 9, 255],
    [8, 255, 214], [7, 255, 224], [255, 184, 6], [10, 255, 71], [255, 41, 10],
    [7, 255, 255], [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
    [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153], [6, 51, 255],
    [235, 12, 255], [160, 150, 20], [0, 163, 255], [140, 140, 140], [250, 10, 15],
    [20, 255, 0], [31, 255, 0], [255, 31, 0], [255, 224, 0], [153, 255, 0],
    [0, 0, 255], [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
    [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255], [0, 255, 112],
    [0, 255, 133], [255, 0, 0], [255, 163, 0], [255, 102, 0], [194, 255, 0],
    [0, 143, 255], [51, 255, 0], [0, 82, 255], [0, 255, 41], [0, 255, 173],
    [10, 0, 255], [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
    [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20], [255, 184, 184],
    [0, 31, 255], [0, 255, 61], [0, 71, 255], [255, 0, 204], [0, 255, 194],
    [0, 255, 82], [0, 10, 255], [0, 112, 255], [51, 0, 255], [0, 194, 255],
    [0, 122, 255], [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
    [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0], [8, 184, 170],
    [133, 0, 255], [0, 255, 92], [184, 0, 255], [255, 0, 31], [0, 184, 255],
    [0, 214, 255], [255, 0, 112], [92, 255, 0], [0, 224, 255], [112, 224, 255],
    [70, 184, 160], [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
    [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0], [255, 0, 235],
    [245, 0, 255], [255, 0, 122], [255, 245, 0], [10, 190, 212], [214, 255, 0],
    [0, 204, 255], [20, 0, 255], [255, 255, 0], [0, 153, 255], [0, 41, 255],
    [0, 255, 204], [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
    [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255], [184, 255, 0],
    [0, 133, 255], [255, 214, 0], [25, 194, 194], [102, 255, 0], [92, 0, 255]
]

def main():
    """
    Démonstration rapide avec segmentation d'une image
    Utilise ViT Small (27M paramètres, 384-dim)
    """
    print("="*80)
    print("DEMO - Vision Transformer for Semantic Segmentation")
    print("Architecture: ViT Small (27M params)")
    print("="*80)
    
    # Vérifier checkpoint
    checkpoint_path = "checkpoint.pth"
    if not os.path.exists(checkpoint_path):
        print("\n❌ ERREUR: checkpoint.pth introuvable!")
        print("\n📥 Pour utiliser ce script, placez checkpoint.pth à la racine")
        print("\n💡 Ou utilisez le script complet:")
        print("   python Scripts/predict.py --image_path IMAGE --checkpoint_path CHECKPOINT")
        return
    
    # Trouver une image de test
    image_path = None
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Chercher dans results/samples
        test_images = list(Path("results/samples").glob("*.png")) + \
                     list(Path("results/samples").glob("*.jpg")) + \
                     list(Path("results/samples").glob("*.jpeg"))
        if test_images:
            image_path = str(test_images[0])
    
    if not image_path or not os.path.exists(image_path):
        print("\n⚠️  Aucune image spécifiée!")
        print("\n💡 Utilisation:")
        print("   python demo.py chemin/vers/image.jpg")
        print("\n   OU placez des images dans results/samples/")
        return
    
    print(f"\n✅ Checkpoint: {checkpoint_path}")
    print(f"✅ Image: {image_path}")
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✅ Device: {device}")
    
    # Charger checkpoint
    print("\n[1/4] Chargement du checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Configuration ViT Small
    if "model_cfg" in checkpoint:
        model_cfg = checkpoint["model_cfg"]
        print("    → Configuration depuis checkpoint")
    else:
        # Configuration ViT Small (par défaut)
        model_cfg = {
            "backbone": "vit_small_patch16_384",
            "image_size": [512, 512],
            "patch_size": 16,
            "d_model": 384,
            "n_heads": 6,
            "n_layers": 12,
            "decoder": {
                "name": "mask_transformer",
                "n_layers": 2,
                "drop_path_rate": 0.0,
                "dropout": 0.1
            },
            "n_cls": 150
        }
        print("    → Configuration ViT Small (384-dim, 6 heads, 12 layers)")
    
    # Créer modèle
    print("[2/4] Création du modèle...")
    model = create_segmenter(model_cfg)
    
    print(f"    → Architecture: ViT Small")
    print(f"    → Paramètres: d_model={model_cfg['d_model']}, heads={model_cfg['n_heads']}, layers={model_cfg['n_layers']}")
    print(f"    → Decoder: Mask Transformer ({model_cfg['decoder']['n_layers']} layers)")
    
    # Charger weights (EMA si disponible)
    if "ema_model" in checkpoint:
        print("    → Chargement modèle EMA")
        model.load_state_dict(checkpoint["ema_model"], strict=True)
    else:
        print("    → Chargement modèle standard")
        model.load_state_dict(checkpoint["model"], strict=True)
    
    model.to(device)
    model.eval()
    
    # Prétraiter image
    print("[3/4] Prétraitement de l'image...")
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inférence
    print("[4/4] Segmentation en cours...")
    with torch.no_grad():
        seg_map = model(input_tensor)
        seg_map = seg_map.argmax(1).cpu().numpy()[0]
    
    # Visualiser
    print("\n✅ Segmentation terminée!")
    
    palette = [c for rgb in ADE20K_PALETTE for c in rgb]
    seg_image = Image.fromarray(seg_map.astype('uint8'))
    seg_image.putpalette(palette)
    seg_image_resized = seg_image.resize(original_size, Image.NEAREST)
    seg_colored = seg_image_resized.convert('RGB')
    overlay_image = Image.blend(image, seg_colored, alpha=0.5)
    
    # Sauvegarder
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_seg = f"{base_name}_demo_segmentation.png"
    output_overlay = f"{base_name}_demo_overlay.png"
    
    seg_colored.save(output_seg)
    overlay_image.save(output_overlay)
    
    print(f"\n💾 Résultats sauvegardés:")
    print(f"   • {output_seg}")
    print(f"   • {output_overlay}")
    
    # Afficher
    print("\n📊 Affichage des résultats...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title("Image Originale")
    axes[0].axis('off')
    
    axes[1].imshow(seg_colored)
    axes[1].set_title("Segmentation")
    axes[1].axis('off')
    
    axes[2].imshow(overlay_image)
    axes[2].set_title("Overlay (50%)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*80)
    print("✅ Démo terminée!")
    print("="*80)

if __name__ == "__main__":
    main()
