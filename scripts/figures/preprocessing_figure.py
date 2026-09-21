"""Show what the two evaluation protocols actually feed a network.

The thesis reports the resize-vs-crop difference at 2.673 dB, 95% CI [1.062, 4.039] — larger
than any training-time effect measured. That number is abstract until you see the inputs:
a 2708x1800 frame squashed to 256^2 loses roughly a tenfold linear scale of detail, while a
256^2 crop keeps native detail over 1.3% of the scene. They are not the same task.
"""
import os, glob, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

SRC = sys.argv[1] if len(sys.argv) > 1 else "/home/amir-fx507/thesis_export/sample_frame.jpg"
OUT = "/home/amir-fx507/thesis_work/figures/preprocessing_protocols"
S = 256

im = Image.open(SRC).convert("RGB")
W, H = im.size
resized = im.resize((S, S), Image.BILINEAR)
top, left = (H - S) // 2, (W - S) // 2
cropped = im.crop((left, top, left + S, top + S))

fig = plt.figure(figsize=(14.5, 6.4))
gs = fig.add_gridspec(2, 3, width_ratios=[1.55, 1, 1], height_ratios=[1, 1],
                      hspace=0.28, wspace=0.18)

# full frame with the crop window marked
ax0 = fig.add_subplot(gs[:, 0])
ax0.imshow(np.asarray(im))
ax0.add_patch(Rectangle((left, top), S, S, fill=False, edgecolor="#c53030", linewidth=2.5))
ax0.set_title(f"Native frame  {W} x {H}\nred box = the 256 x 256 crop region", fontsize=11)
ax0.axis("off")

ax1 = fig.add_subplot(gs[0, 1])
ax1.imshow(np.asarray(resized)); ax1.axis("off")
ax1.set_title("Protocol A — resize whole frame\nentire scene, ~10x downscale", fontsize=10.5)

ax2 = fig.add_subplot(gs[0, 2])
ax2.imshow(np.asarray(cropped)); ax2.axis("off")
ax2.set_title("Protocol B — crop at native resolution\n~1.3% of scene, full detail", fontsize=10.5)

# zoom into the same 64px region of each, to show the detail difference
z = 64
ra = np.asarray(resized)[S//2-z//2:S//2+z//2, S//2-z//2:S//2+z//2]
cb = np.asarray(cropped)[S//2-z//2:S//2+z//2, S//2-z//2:S//2+z//2]
ax3 = fig.add_subplot(gs[1, 1]); ax3.imshow(ra, interpolation="nearest"); ax3.axis("off")
ax3.set_title("centre 64 x 64 of A", fontsize=9.5)
ax4 = fig.add_subplot(gs[1, 2]); ax4.imshow(cb, interpolation="nearest"); ax4.axis("off")
ax4.set_title("centre 64 x 64 of B", fontsize=9.5)

fig.suptitle("The two evaluation protocols are different tasks — measured at 2.673 dB, 95 % CI [1.062, 4.039]",
             fontsize=12.5, y=0.99)
fig.savefig(f"{OUT}.png", dpi=160, bbox_inches="tight")
fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
print(f"wrote {OUT}.png / .pdf   (source {W}x{H})")
