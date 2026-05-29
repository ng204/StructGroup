"""
Draw StructGroup architecture diagram
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch, Rectangle
import numpy as np

# Set font
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

fig = plt.figure(figsize=(22, 16))
ax = fig.add_subplot(111)
ax.set_xlim(0, 22)
ax.set_ylim(0, 16)
ax.axis('off')

# Define colors
color_backbone = '#E0E0E0'  # Gray
color_boundary = '#4A90E2'  # Blue
color_junction = '#50C878'  # Green
color_adaptive = '#FF8C42'  # Orange
color_loss = '#FF6B6B'  # Red
color_output = '#9B59B6'  # Purple
color_bg = '#F5F5F5'  # Light gray background

# Add background regions
def draw_background_region(ax, x, y, width, height, color, alpha=0.3):
    """Draw background region"""
    rect = Rectangle((x-width/2, y-height/2), width, height,
                    facecolor=color, edgecolor='none', alpha=alpha, zorder=0)
    ax.add_patch(rect)

# Function: Draw rounded text box
def draw_box(ax, x, y, width, height, text, color, text_size=9, bold=False, alpha=1.0):
    """Draw rounded text box"""
    box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                        boxstyle="round,pad=0.15", 
                        facecolor=color, edgecolor='black', linewidth=1.8, alpha=alpha)
    ax.add_patch(box)
    
    fontweight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', 
           fontsize=text_size, fontweight=fontweight, wrap=True)

# Function: Draw arrow with better styling
def draw_arrow(ax, x1, y1, x2, y2, color='black', style='->', linewidth=2.0, alpha=0.8):
    """Draw arrow"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style, color=color, 
                           linewidth=linewidth, zorder=2, alpha=alpha,
                           connectionstyle="arc3,rad=0.1")
    ax.add_patch(arrow)

# Function: Draw straight arrow
def draw_straight_arrow(ax, x1, y1, x2, y2, color='black', linewidth=2.0):
    """Draw straight vertical arrow"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', color=color, 
                           linewidth=linewidth, zorder=2)
    ax.add_patch(arrow)

# ========== Title ==========
ax.text(11, 15.5, 'StructGroup Architecture', ha='center', va='center', 
       fontsize=20, fontweight='bold')

# ========== Background regions for better visual separation ==========
draw_background_region(ax, 11, 10.5, 20, 3, color_bg, alpha=0.2)  # Feature enhancement region
draw_background_region(ax, 11, 6.5, 20, 2, color_bg, alpha=0.2)  # Grouping region
draw_background_region(ax, 11, 3, 20, 3, color_bg, alpha=0.2)  # Instance adaptive region

# ========== Layer 1: Input ==========
draw_box(ax, 11, 14, 3.5, 0.9, 'Input Point Cloud\n(N points)', color_backbone, bold=True, text_size=10)

# ========== Layer 2: Backbone ==========
draw_box(ax, 11, 12.5, 4.5, 1.1, 'Backbone: U-Net\nPoint Features (N, C)', color_backbone, bold=True, text_size=10)
draw_straight_arrow(ax, 11, 13.1, 11, 12.95)

# ========== Layer 3: Improvement Modules (Side by side) ==========
# Module 1 - Boundary Attention (Left)
draw_box(ax, 4, 10.5, 3.8, 3, 'Module 1:\nBoundary Attention\n[MLP Network]', color_boundary, bold=True, text_size=11)

# Boundary attention internal structure (better spacing)
draw_box(ax, 4, 9.6, 3.4, 0.7, 'Multi-scale Geometry\nFeatures (k=8,16,32)', color_boundary, text_size=9)
draw_box(ax, 4, 8.7, 3.4, 0.7, 'Boundary Detection\n[MLP: C+24→256→1]', color_boundary, text_size=9)
draw_box(ax, 4, 7.8, 3.4, 0.7, 'Feature Enhancement\n+ Semantic Fusion', color_boundary, text_size=9)

# Module 2 - Junction Attention (Right)
draw_box(ax, 18, 10.5, 3.8, 3, 'Module 2:\nJunction Attention\n[MLP Network]', color_junction, bold=True, text_size=11)

# Junction attention internal structure
draw_box(ax, 18, 9.6, 3.4, 0.7, 'Semantic Ambiguity\n(stem×leaf, stem×branch)', color_junction, text_size=9)
draw_box(ax, 18, 8.7, 3.4, 0.7, 'Rich Geometry Features\n(9-dim)', color_junction, text_size=9)
draw_box(ax, 18, 7.8, 3.4, 0.7, 'Junction Detection\n+ Direction Prediction', color_junction, text_size=9)

# Arrows from backbone to modules
draw_arrow(ax, 11, 12, 6.2, 10.5, color_boundary, linewidth=2.5, alpha=0.9)
draw_arrow(ax, 11, 12, 15.8, 10.5, color_junction, linewidth=2.5, alpha=0.9)

# Arrows from modules to semantic/offset
draw_arrow(ax, 6.2, 10.5, 11, 9.5, color_boundary, linewidth=2.5, alpha=0.9)
draw_arrow(ax, 15.8, 10.5, 11, 9.5, color_junction, linewidth=2.5, alpha=0.9)

# ========== Layer 4: Semantic and Offset Prediction ==========
draw_box(ax, 11, 9.5, 5, 1.1, 'Semantic Prediction\n+ Offset Prediction', color_backbone, bold=True, text_size=10)
draw_straight_arrow(ax, 11, 10.5, 11, 10.05)

# ========== Layer 5: Grouping ==========
draw_box(ax, 11, 7.5, 5, 1.1, 'Grouping: BFS Clustering\nProposals (K instances)', color_backbone, bold=True, text_size=10)
draw_straight_arrow(ax, 11, 9, 11, 8.05)

# ========== Layer 6: Instance Head ==========
draw_box(ax, 11, 5.5, 5, 1.1, 'Instance Head\nClassification + Mask + IoU', color_backbone, bold=True, text_size=10)
draw_straight_arrow(ax, 11, 7, 11, 6.05)

# ========== Layer 7: Improvement Module 3 - Instance Adaptive ==========
draw_box(ax, 11, 3, 6, 2.8, 'Module 3:\nInstance Adaptive\n[MLP Network]', color_adaptive, bold=True, text_size=11)

# Instance adaptive internal structure (better horizontal spacing)
draw_box(ax, 7.5, 2.2, 2.8, 0.7, 'Proposal Feature\nExtraction\n[Geometric+Statistical]', color_adaptive, text_size=9)
draw_box(ax, 11, 2.2, 2.8, 0.7, 'Temperature Scaling\n+ Logit Bias\n[MLP]', color_adaptive, text_size=9)
draw_box(ax, 14.5, 2.2, 2.8, 0.7, 'Reliability\nPrediction\n[MLP]', color_adaptive, text_size=9)

draw_straight_arrow(ax, 11, 5, 11, 4.4)
draw_straight_arrow(ax, 11, 1.6, 11, 1.1)

# ========== Layer 8: NMS and Post-processing ==========
draw_box(ax, 11, 1, 5, 1.1, 'NMS + Post-processing\nFinal Instance Results', color_output, bold=True, text_size=10)
draw_straight_arrow(ax, 11, 1.6, 11, 1.55)

# ========== Loss function annotations (better positioned) ==========
# Boundary Loss (left side, aligned with Module 1)
draw_box(ax, 0.8, 9.6, 1.5, 0.6, 'Boundary\nAware Loss', color_loss, text_size=8)
draw_arrow(ax, 1.55, 9.6, 2.2, 9.6, color_loss, style='->', linewidth=2, alpha=0.8)

# Junction Loss (right side, aligned with Module 2)
draw_box(ax, 21.2, 9.6, 1.5, 0.6, 'Junction\nAware Loss', color_loss, text_size=8)
draw_arrow(ax, 20.45, 9.6, 19.8, 9.6, color_loss, style='->', linewidth=2, alpha=0.8)

# Reliability Loss (left side, aligned with Module 3)
draw_box(ax, 0.8, 2.2, 1.5, 0.6, 'Reliability\nLoss', color_loss, text_size=8)
draw_arrow(ax, 1.55, 2.2, 4.7, 2.2, color_loss, style='->', linewidth=2, alpha=0.8)

# ========== Add connection lines to show data flow ==========
# Vertical flow line in center
center_x = 11
for y in [13.1, 12.95, 10.05, 8.05, 6.05, 4.4, 1.55]:
    if y < 13.1:
        draw_straight_arrow(ax, center_x, y+0.15, center_x, y, 'gray', linewidth=1.5)

# ========== Legend (better positioned) ==========
legend_elements = [
    mpatches.Patch(facecolor=color_backbone, edgecolor='black', label='Original SoftGroup Modules'),
    mpatches.Patch(facecolor=color_boundary, edgecolor='black', label='Module 1: Boundary Attention'),
    mpatches.Patch(facecolor=color_junction, edgecolor='black', label='Module 2: Junction Attention'),
    mpatches.Patch(facecolor=color_adaptive, edgecolor='black', label='Module 3: Instance Adaptive'),
    mpatches.Patch(facecolor=color_loss, edgecolor='black', label='Loss Functions'),
    mpatches.Patch(facecolor=color_output, edgecolor='black', label='Output Results'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.95, 
         edgecolor='black', fancybox=True, shadow=True)

# Save figure
output_path = '/home/ng204/xueshuang/SoftGroup/vis_output/structgroup_architecture_diagram.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Architecture diagram saved to: {output_path}")

plt.show()
