"""Publication-quality architecture diagrams for TRDN and VideoEENet.
Vector PDF + raster PNG. Frozen components are visually distinguished from trainable ones,
and every block carries its measured parameter count."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.patheffects as pe

OUT = "/home/amir-fx507/thesis_work/figures"

TRAIN = "#2b6cb0"    # trainable
FROZEN = "#718096"   # frozen
DATA   = "#2f855a"   # tensors / data
ACCENT = "#c53030"   # the finding worth pointing at

def box(ax, x, y, w, h, label, sub=None, color=TRAIN, fc=None, fontsize=9.5, hatch=None):
    fc = fc or color
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                       linewidth=1.5, edgecolor=color, facecolor=fc, alpha=0.16 if hatch is None else 0.10,
                       hatch=hatch, zorder=2)
    ax.add_patch(p)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.5, edgecolor=color, facecolor="none", zorder=3))
    ax.text(x + w/2, y + h/2 + (0.030 if sub else 0), label, ha="center", va="center",
            fontsize=fontsize, color="#1a202c", zorder=4, weight="semibold")
    if sub:
        ax.text(x + w/2, y + h/2 - 0.045, sub, ha="center", va="center",
                fontsize=fontsize - 2.1, color="#4a5568", zorder=4, style="italic")
    return (x + w/2, y)

def arrow(ax, p1, p2, color="#4a5568", style="-|>", lw=1.5, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=13,
                                 linewidth=lw, color=color, zorder=1, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=2, shrinkB=2))

# ============================ TRDN ============================
fig, ax = plt.subplots(figsize=(13.2, 7.4))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

ax.text(0.5, 0.965, "TRDN — Temporal Reference-Guided Diffusion Network",
        ha="center", fontsize=14.5, weight="bold", color="#1a202c")
ax.text(0.5, 0.925, "864,449,287 trainable parameters   •   frozen components shown hatched",
        ha="center", fontsize=9.5, color="#4a5568")

# input
box(ax, 0.02, 0.70, 0.155, 0.135, "Hazy sequence", "T = 10 frames, 256×256", DATA)
box(ax, 0.02, 0.50, 0.155, 0.125, "Target frame", "frame t", DATA)

# stage 1 RAFT
box(ax, 0.215, 0.70, 0.15, 0.135, "RAFT", "optical flow · FROZEN", FROZEN, hatch="////")
# stage 2 ConvLSTM
box(ax, 0.405, 0.70, 0.15, 0.135, "ConvLSTM\nmemory", "371 K", TRAIN)
# stage 3 transformer
box(ax, 0.595, 0.70, 0.16, 0.135, "Retrieval\ntransformer", "3.3 M", TRAIN)
# stage 4 selector  -- the finding
box(ax, 0.795, 0.70, 0.175, 0.135, "Reference\nselector", "154 K", ACCENT)

arrow(ax, (0.175, 0.767), (0.215, 0.767))
arrow(ax, (0.365, 0.767), (0.405, 0.767))
arrow(ax, (0.555, 0.767), (0.595, 0.767))
arrow(ax, (0.755, 0.767), (0.795, 0.767))

# selector finding callout
ax.text(0.883, 0.672, "measured: weight 1.000 on offset −1,\n0.000 on offsets −2 … −9",
        ha="center", va="top", fontsize=8.3, color=ACCENT, style="italic")

# conditioning adapter — sits directly above the UNet, clear of the decoder column
box(ax, 0.345, 0.455, 0.17, 0.115, "Conditioning\nadapter", "1.08 M", TRAIN)
arrow(ax, (0.883, 0.70), (0.515, 0.535), rad=0.22)

# UNet
box(ax, 0.30, 0.245, 0.26, 0.175, "Stable Diffusion inpainting UNet",
    "859.5 M · fully fine-tuned · latent space", TRAIN, fontsize=10.5)
arrow(ax, (0.43, 0.455), (0.43, 0.42))

# VAE encode/decode
box(ax, 0.055, 0.245, 0.14, 0.175, "VAE\nencoder", "FROZEN", FROZEN, hatch="////")
box(ax, 0.665, 0.245, 0.14, 0.175, "VAE\ndecoder", "FROZEN", FROZEN, hatch="////")
arrow(ax, (0.195, 0.332), (0.30, 0.332))
arrow(ax, (0.56, 0.332), (0.665, 0.332))
arrow(ax, (0.0975, 0.50), (0.0975, 0.42))

# DDIM
ax.text(0.43, 0.208, "50 DDIM steps, guidance 1.0", ha="center", fontsize=8.6,
        color="#4a5568", style="italic")

# output + ceiling
box(ax, 0.845, 0.245, 0.135, 0.175, "Dehazed\nframe", "21.78 dB", DATA)
arrow(ax, (0.805, 0.332), (0.845, 0.332))

ax.add_patch(Rectangle((0.055, 0.055), 0.925, 0.105, facecolor="#fff5f5",
                       edgecolor=ACCENT, linewidth=1.2, alpha=0.5, zorder=0))
ax.text(0.5175, 0.128, "Frozen decoder imposes a hard ceiling of 34.566 dB (measured through the identical harness)",
        ha="center", fontsize=9.3, color=ACCENT, weight="semibold")
ax.text(0.5175, 0.084, "Temporal machinery (memory + transformer + selector) = 0.45 % of trainable weights;  the remaining 99.55 % is the adapted image prior",
        ha="center", fontsize=8.6, color="#4a5568")

fig.savefig(f"{OUT}/architecture_trdn.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/architecture_trdn.png", dpi=190, bbox_inches="tight")
plt.close(fig)
print("wrote architecture_trdn")

# ============================ VideoEENet ============================
fig, ax = plt.subplots(figsize=(13.2, 5.9))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

ax.text(0.5, 0.955, "VideoEENet — Recurrent Dual-Domain Video Dehazer",
        ha="center", fontsize=14.5, weight="bold", color="#1a202c")
ax.text(0.5, 0.905, "1,927,000 trainable parameters   •   base_channels 32, hidden_dim 64, 4 attention heads",
        ha="center", fontsize=9.5, color="#4a5568")

box(ax, 0.02, 0.60, 0.145, 0.185, "Hazy sequence", "T = 10 · 256×256", DATA)
box(ax, 0.205, 0.60, 0.165, 0.185, "Dual-domain\nencoder", "0.977 M", TRAIN)
box(ax, 0.41, 0.60, 0.155, 0.185, "ConvLSTM", "0.443 M", TRAIN)
box(ax, 0.605, 0.60, 0.155, 0.185, "Temporal\nattention", "0.113 M · 4 heads", TRAIN)
box(ax, 0.80, 0.60, 0.145, 0.185, "Decoder", "0.386 M", TRAIN)

for a, b in ((0.165, 0.205), (0.37, 0.41), (0.565, 0.605), (0.76, 0.80)):
    arrow(ax, (a, 0.692), (b, 0.692))

box(ax, 0.80, 0.335, 0.145, 0.165, "Clean\nframe t", "21.53 dB", DATA)
arrow(ax, (0.8725, 0.60), (0.8725, 0.50))

ax.text(0.30, 0.44, "LayerNorm ×5, no BatchNorm\n(batch size 1 makes batch statistics meaningless)",
        ha="center", fontsize=9, color="#4a5568", style="italic")
ax.text(0.30, 0.335, "Predicts the FINAL frame of each window\nfrom all T frames; sliding window, stride 1",
        ha="center", fontsize=9, color="#4a5568", style="italic")

ax.add_patch(Rectangle((0.02, 0.055), 0.925, 0.20, facecolor="#f0fff4",
                       edgecolor=DATA, linewidth=1.2, alpha=0.45, zorder=0))
ax.text(0.4825, 0.215, "Augmentation (measured +0.70 / +0.77 dB across two seeds; seed spread 0.067 dB)",
        ha="center", fontsize=10, color="#22543d", weight="semibold")
ax.text(0.4825, 0.163, "variable-scale crop 0.6–1.0   •   horizontal flip p = 0.5   •   temporal reversal p = 0.25",
        ha="center", fontsize=9, color="#2f855a")
ax.text(0.4825, 0.100, "EXCLUDED on physical grounds:  vertical flip (haze varies with depth/height)   •   colour jitter (airlight is read from absolute colour statistics)",
        ha="center", fontsize=8.7, color="#742a2a")

fig.savefig(f"{OUT}/architecture_videoeenet.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/architecture_videoeenet.png", dpi=190, bbox_inches="tight")
plt.close(fig)
print("wrote architecture_videoeenet")
