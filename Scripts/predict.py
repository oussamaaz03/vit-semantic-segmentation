
import os
import yaml
import argparse
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from segm.model.factory import create_segmenter
from segm.utils.ema import ModelEMA

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
    parser = argparse.ArgumentParser(description="Script de prédiction pour la segmentation d'images.")
    parser.add_argument('--image_path', type=str, required=True, help='Chemin vers l\'image à segmenter.')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint.pth', help='Chemin vers le fichier checkpoint du modèle.')
    parser.add_argument('--config_path', type=str, default='segm/config.yml', help='Chemin vers le fichier de configuration du modèle.')
    parser.add_argument('--output_dir', type=str, default='.', help='Dossier où enregistrer l\'image segmentée.')
    parser.add_argument('--use_ema', action='store_true', default=True, help='Utiliser le modèle EMA (meilleure performance)')
    parser.add_argument('--use_tta', action='store_true', help='Utiliser Test-Time Augmentation pour améliorer la précision')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cuda/cpu)')
    args = parser.parse_args()

    print(f"[INFO] Chargement du checkpoint: {args.checkpoint_path}")
    if not os.path.exists(args.checkpoint_path):
        print(f"[ERROR] Le fichier checkpoint '{args.checkpoint_path}' n'a pas été trouvé.")
        return

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")

    # Récupérer la config depuis le checkpoint si disponible
    if "model_cfg" in checkpoint:
        print("[INFO] Utilisation de la configuration du checkpoint...")
        model_cfg = checkpoint["model_cfg"]
    else:
        # Détecter la taille du modèle depuis les weights
        print("[WARN] Configuration non trouvée dans le checkpoint.")
        print("[INFO] Détection automatique de la taille du modèle...")
        
        # Détecter d_model depuis encoder.cls_token shape
        if "model" in checkpoint:
            d_model = checkpoint["model"]["encoder.cls_token"].shape[-1]
        elif "ema_model" in checkpoint:
            d_model = checkpoint["ema_model"]["encoder.cls_token"].shape[-1]
        else:
            d_model = 192  # Fallback
        
        # Déterminer le backbone
        if d_model == 192:
            backbone = "vit_tiny_patch16_384"
            n_heads = 3
            print(f"[INFO] Détecté: ViT Tiny (d_model={d_model})")
        elif d_model == 384:
            backbone = "vit_small_patch16_384"
            n_heads = 6
            print(f"[INFO] Détecté: ViT Small (d_model={d_model})")
        elif d_model == 768:
            backbone = "vit_base_patch16_384"
            n_heads = 12
            print(f"[INFO] Détecté: ViT Base (d_model={d_model})")
        else:
            backbone = "vit_small_patch16_384"
            n_heads = 6
            print(f"[WARN] Taille inconnue (d_model={d_model}), utilisation de ViT Small par défaut")
        
        model_cfg = {
            "backbone": backbone,
            "image_size": [512, 512],
            "patch_size": 16,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": 12,
            "normalization": "vit",
            "distilled": False,
            "decoder": {
                "name": "mask_transformer",
                "drop_path_rate": 0.0,
                "dropout": 0.1,
                "n_layers": 2
            },
            "n_cls": 150  # ADE20K a 150 classes
        }

    # Création du modèle
    print("[INFO] Création du modèle...")
    model = create_segmenter(model_cfg)
    
    # Charger les weights appropriés
    if args.use_ema and "ema_model" in checkpoint:
        print("[INFO] Utilisation du modèle EMA (meilleure performance)...")
        model.load_state_dict(checkpoint["ema_model"], strict=True)
    else:
        print("[INFO] Utilisation du modèle standard...")
        model.load_state_dict(checkpoint["model"], strict=True)
    
    model.to(args.device)
    model.eval()

    # Prétraitement de l'image
    print(f"[INFO] Prétraitement de l'image: {args.image_path}")
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(args.image_path).convert("RGB")
    original_size = image.size  # Sauvegarder taille originale
    input_tensor = transform(image).unsqueeze(0).to(args.device)

    # Inférence (avec ou sans TTA)
    if args.use_tta:
        print("[INFO] Exécution de l'inférence avec Test-Time Augmentation (TTA)...")
        print("[INFO] TTA: Cette méthode améliore la précision de ~1-2% mais prend 4× plus de temps")
        
        with torch.no_grad():
            predictions = []
            
            # 1. Original
            print("[TTA] 1/4 - Image originale...")
            seg_map_1 = model(input_tensor)
            predictions.append(seg_map_1)
            
            # 2. Flip horizontal
            print("[TTA] 2/4 - Flip horizontal...")
            input_flipped = torch.flip(input_tensor, dims=[3])  # Flip sur dimension width
            seg_map_2 = model(input_flipped)
            seg_map_2 = torch.flip(seg_map_2, dims=[3])  # Re-flip la prédiction
            predictions.append(seg_map_2)
            
            # 3. Scale 0.75×
            print("[TTA] 3/4 - Scale 0.75×...")
            transform_small = transforms.Compose([
                transforms.Resize((int(384 * 0.75), int(384 * 0.75))),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_small = transform_small(image).unsqueeze(0).to(args.device)
            seg_map_3 = model(input_small)
            seg_map_3 = torch.nn.functional.interpolate(seg_map_3, size=(384, 384), mode='bilinear', align_corners=False)
            predictions.append(seg_map_3)
            
            # 4. Scale 1.25×
            print("[TTA] 4/4 - Scale 1.25×...")
            transform_large = transforms.Compose([
                transforms.Resize((int(384 * 1.25), int(384 * 1.25))),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_large = transform_large(image).unsqueeze(0).to(args.device)
            seg_map_4 = model(input_large)
            seg_map_4 = torch.nn.functional.interpolate(seg_map_4, size=(384, 384), mode='bilinear', align_corners=False)
            predictions.append(seg_map_4)
            
            # Moyenne des 4 prédictions
            print("[INFO] Fusion des 4 prédictions...")
            seg_map = torch.stack(predictions).mean(0)
            seg_map = seg_map.argmax(1).cpu().numpy()[0]
            
            print("[SUCCESS] TTA terminé! Précision améliorée de ~1-2%")
    else:
        print("[INFO] Exécution de l'inférence standard (sans TTA)...")
        print("[TIP] Utilisez --use_tta pour améliorer la précision (+1-2% Mean IoU)")
        
        with torch.no_grad():
            seg_map = model(input_tensor)
            seg_map = seg_map.argmax(1).cpu().numpy()[0]

    # Visualisation et enregistrement
    print("[INFO] Création des visualisations...")
    # Flatten palette for PIL
    palette = [c for rgb in ADE20K_PALETTE for c in rgb]
    
    # Créer une image couleur à partir de la carte de segmentation
    seg_image = Image.fromarray(seg_map.astype('uint8'))
    seg_image.putpalette(palette)
    
    # Redimensionner à la taille originale
    seg_image_resized = seg_image.resize(original_size, Image.NEAREST)
    
    # Créer 3 versions: segmentation seule, overlay, et côte-à-côte
    # 1. Segmentation colorée
    seg_colored = seg_image_resized.convert('RGB')
    
    # 2. Overlay (blend avec image originale)
    overlay_image = Image.blend(image, seg_colored, alpha=0.5)
    
    # 3. Côte-à-côte
    combined_width = original_size[0] * 2
    combined_image = Image.new('RGB', (combined_width, original_size[1]))
    combined_image.paste(image, (0, 0))
    combined_image.paste(seg_colored, (original_size[0], 0))
    
    # Sauvegarder les 3 versions
    base_name = os.path.splitext(os.path.basename(args.image_path))[0]
    output_seg = os.path.join(args.output_dir, f"{base_name}_segmentation.png")
    output_overlay = os.path.join(args.output_dir, f"{base_name}_overlay.png")
    output_combined = os.path.join(args.output_dir, f"{base_name}_combined.png")
    
    seg_colored.save(output_seg)
    overlay_image.save(output_overlay)
    combined_image.save(output_combined)
    
    print(f"[SUCCESS] Résultats sauvegardés:")
    print(f"  - Segmentation: {output_seg}")
    print(f"  - Overlay: {output_overlay}")
    print(f"  - Comparaison: {output_combined}")

    # Afficher les résultats
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


if __name__ == '__main__':
    main()
