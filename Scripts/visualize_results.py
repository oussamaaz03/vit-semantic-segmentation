#!/usr/bin/env python
"""
Visualisations à partir du fichier evaluation_complete_2000img.txt
Crée des graphiques et heatmaps sans re-exécuter l'évaluation
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Lire le fichier de résultats
results_file = "evaluation_complete_2000img.txt"

print("=" * 80)
print("CRÉATION DE VISUALISATIONS À PARTIR DES RÉSULTATS EXISTANTS")
print("=" * 80)

# Parser le fichier
print(f"\n[1/4] Lecture de {results_file}...")
class_names = []
class_ious = []
class_counts = []

with open(results_file, 'r', encoding='utf-8') as f:
    content = f.read()
    
# Extraire les données des classes
lines = content.split('\n')
parsing_classes = False

for line in lines:
    if 'mIoU PAR CLASSE' in line:
        parsing_classes = True
        continue
    
    if parsing_classes and line.strip().startswith(tuple('0123456789')):
        # Format: "   0. wall                 → 58.93% (1167 imgs)"
        parts = line.strip().split('→')
        if len(parts) == 2:
            # Partie gauche: "0. wall"
            left = parts[0].strip()
            class_id = int(left.split('.')[0])
            class_name = '.'.join(left.split('.')[1:]).strip()
            
            # Partie droite: "58.93% (1167 imgs)"
            right = parts[1].strip()
            iou = float(right.split('%')[0])
            count = int(right.split('(')[1].split(' ')[0])
            
            class_names.append(class_name)
            class_ious.append(iou)
            class_counts.append(count)

class_ious = np.array(class_ious)
class_counts = np.array(class_counts)

print(f"✓ {len(class_names)} classes extraites")

# Créer dossier de sortie
output_dir = Path("visualizations")
output_dir.mkdir(exist_ok=True)

# ============================================================================
# VISUALISATION 1: Heatmap 10×15 des IoUs
# ============================================================================
print("\n[2/4] Création heatmap IoU par classe...")

fig, ax = plt.subplots(figsize=(20, 14))

# Reshape en grille 10×15 pour les 150 classes
iou_matrix = class_ious.reshape(10, 15)

# Heatmap
im = ax.imshow(iou_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

# Annotations avec noms de classes
for i in range(10):
    for j in range(15):
        class_idx = i * 15 + j
        text_color = 'white' if iou_matrix[i, j] < 50 else 'black'
        ax.text(j, i, f"{class_names[class_idx][:8]}\n{iou_matrix[i, j]:.1f}%",
                ha="center", va="center", color=text_color, fontsize=8)

# Colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('IoU (%)', fontsize=14)

ax.set_title('Heatmap IoU - 150 Classes ADE20K (10×15)', fontsize=16, weight='bold', pad=20)
ax.set_xlabel('Classe (colonne)', fontsize=12)
ax.set_ylabel('Classe (ligne)', fontsize=12)
ax.set_xticks(range(15))
ax.set_yticks(range(10))
ax.set_xticklabels(range(15))
ax.set_yticklabels(range(10))

plt.tight_layout()
plt.savefig(output_dir / "heatmap_iou_150classes.png", dpi=200, bbox_inches='tight')
print(f"✓ Sauvegardé: {output_dir / 'heatmap_iou_150classes.png'}")
plt.close()

# ============================================================================
# VISUALISATION 2: Bar chart IoU toutes les classes
# ============================================================================
print("\n[3/4] Création graphique bars IoU...")

fig, ax = plt.subplots(figsize=(28, 8))

# Couleurs selon performance
colors = ['red' if iou < 20 else 'orange' if iou < 40 else 'lightgreen' if iou < 60 else 'green' 
          for iou in class_ious]

bars = ax.bar(range(150), class_ious, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

# Ligne moyenne
mean_iou = class_ious.mean()
ax.axhline(y=mean_iou, color='blue', linestyle='--', linewidth=2, 
           label=f'Moyenne: {mean_iou:.2f}%')

ax.set_xlabel('Classe ID', fontsize=14, weight='bold')
ax.set_ylabel('IoU (%)', fontsize=14, weight='bold')
ax.set_title('IoU par Classe - 150 Classes ADE20K', fontsize=16, weight='bold')
ax.set_xlim(-1, 150)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.legend(fontsize=12)

# Annotations pour top 5
top5_indices = np.argsort(class_ious)[-5:]
for idx in top5_indices:
    ax.text(idx, class_ious[idx] + 2, class_names[idx][:6], 
            ha='center', fontsize=8, rotation=90)

plt.tight_layout()
plt.savefig(output_dir / "bar_chart_iou_all_classes.png", dpi=200, bbox_inches='tight')
print(f"✓ Sauvegardé: {output_dir / 'bar_chart_iou_all_classes.png'}")
plt.close()

# ============================================================================
# VISUALISATION 3: Distribution IoU et fréquence
# ============================================================================
print("\n[4/4] Création scatter plot IoU vs Fréquence...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Scatter plot: IoU vs Nombre d'apparitions
scatter_colors = class_ious
sc = ax1.scatter(class_counts, class_ious, c=scatter_colors, cmap='RdYlGn', 
                s=100, alpha=0.6, edgecolors='black', linewidth=0.5, vmin=0, vmax=100)
ax1.set_xlabel('Nombre d\'apparitions (images)', fontsize=12, weight='bold')
ax1.set_ylabel('IoU (%)', fontsize=12, weight='bold')
ax1.set_title('IoU vs Fréquence des Classes', fontsize=14, weight='bold')
ax1.grid(alpha=0.3)
ax1.set_xscale('log')
plt.colorbar(sc, ax=ax1, label='IoU (%)')

# Annoter quelques points intéressants
# Best performers
best_idx = np.argmax(class_ious)
ax1.annotate(f'{class_names[best_idx]} ({class_ious[best_idx]:.1f}%)',
             xy=(class_counts[best_idx], class_ious[best_idx]),
             xytext=(10, 10), textcoords='offset points',
             bbox=dict(boxstyle='round', fc='lightgreen', alpha=0.7),
             arrowprops=dict(arrowstyle='->', color='darkgreen'))

# Worst performer (avec occurrences suffisantes)
valid_worst = np.where(class_counts > 20)[0]
if len(valid_worst) > 0:
    worst_idx = valid_worst[np.argmin(class_ious[valid_worst])]
    ax1.annotate(f'{class_names[worst_idx]} ({class_ious[worst_idx]:.1f}%)',
                 xy=(class_counts[worst_idx], class_ious[worst_idx]),
                 xytext=(10, -20), textcoords='offset points',
                 bbox=dict(boxstyle='round', fc='lightcoral', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', color='darkred'))

# Histogramme IoU
ax2.hist(class_ious, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
ax2.axvline(mean_iou, color='red', linestyle='--', linewidth=2, 
            label=f'Moyenne: {mean_iou:.2f}%')
ax2.axvline(np.median(class_ious), color='orange', linestyle='--', linewidth=2, 
            label=f'Médiane: {np.median(class_ious):.2f}%')
ax2.set_xlabel('IoU (%)', fontsize=12, weight='bold')
ax2.set_ylabel('Nombre de classes', fontsize=12, weight='bold')
ax2.set_title('Distribution des IoU', fontsize=14, weight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "scatter_iou_frequency.png", dpi=200, bbox_inches='tight')
print(f"✓ Sauvegardé: {output_dir / 'scatter_iou_frequency.png'}")
plt.close()

# ============================================================================
# RÉSUMÉ
# ============================================================================
print("\n" + "=" * 80)
print("✅ VISUALISATIONS CRÉÉES AVEC SUCCÈS!")
print("=" * 80)
print(f"\n📁 Résultats dans: {output_dir}/")
print(f"   1. heatmap_iou_150classes.png       - Heatmap 10×15 des IoU")
print(f"   2. bar_chart_iou_all_classes.png    - Graphique bars (toutes classes)")
print(f"   3. scatter_iou_frequency.png        - IoU vs Fréquence + Distribution")
print("\n📊 Statistiques rapides:")
print(f"   - Moyenne IoU:  {mean_iou:.2f}%")
print(f"   - Médiane IoU:  {np.median(class_ious):.2f}%")
print(f"   - Min IoU:      {class_ious.min():.2f}% ({class_names[np.argmin(class_ious)]})")
print(f"   - Max IoU:      {class_ious.max():.2f}% ({class_names[np.argmax(class_ious)]})")
print(f"   - Classes > 50%: {(class_ious > 50).sum()}/150")
print(f"   - Classes < 10%: {(class_ious < 10).sum()}/150")
print("=" * 80)
