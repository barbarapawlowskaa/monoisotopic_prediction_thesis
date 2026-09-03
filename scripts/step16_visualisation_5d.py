"""
step16_visualisation_5d.py

Illustration of the five-dimensional orthogonal vector space used in
the thesis. Because we cannot display 5D directly, the figure uses
a 3D projection where the three main axes represent the three physically
motivated basis vectors (V1, V2, V3), and the two null-space vectors
(V4, V5) are shown as additional arrows in that space.

This script is standalone and does not depend on any pipeline data.

Outputs:
    figures/step16_vector_space_5d.png   conceptual diagram for the thesis
"""

import numpy as np
import matplotlib.pyplot as plt

from step00_shared_utils import figures_dir

fig = plt.figure(figsize=(12, 9))
ax  = fig.add_subplot(111, projection="3d")

origin = [0, 0, 0]

vectors = np.array([
    [1,    0,    0   ],   # V1: mass direction
    [0,    1,    0   ],   # V2: variance direction
    [0,    0,    1   ],   # V3: mass defect direction
    [-0.5, 0.5,  0.25],   # V4: null-space correction vector
    [0.3, -0.4,  0.6 ],   # V5: null-space correction vector
])

colors = ["teal", "firebrick", "lightsteelblue", "purple", "pink"]
labels = [
    "V1: average mass",
    "V2: isotopic variance",
    "V3: mass defect",
    "V4: null-space complement",
    "V5: null-space complement",
]

for i in range(5):
    ax.quiver(
        origin[0], origin[1], origin[2],
        vectors[i, 0], vectors[i, 1], vectors[i, 2],
        color=colors[i],
        label=labels[i],
        linewidth=2.5,
        arrow_length_ratio=0.12,
    )

# A schematic protein spectrum point projected onto the basis
spectrum_point = np.array([0.7, 0.8, 0.6])
ax.scatter(
    spectrum_point[0], spectrum_point[1], spectrum_point[2],
    color="black", s=120, zorder=5, label="Protein spectrum vector",
)
ax.plot(
    [0, spectrum_point[0]],
    [0, spectrum_point[1]],
    [0, spectrum_point[2]],
    "k--", alpha=0.4, linewidth=1.2,
)

# Dashed projections onto each axis to emphasise orthogonal decomposition
for axis, point in [
    ([spectrum_point[0], 0, 0], spectrum_point),
    ([0, spectrum_point[1], 0], spectrum_point),
    ([0, 0, spectrum_point[2]], spectrum_point),
]:
    ax.plot(
        [axis[0], point[0]],
        [axis[1], point[1]],
        [axis[2], point[2]],
        color="grey", linestyle=":", alpha=0.5, linewidth=1,
    )

# Schematic constraint plane (the subspace spanned by V1, V2, V3)
xx, yy = np.meshgrid(np.linspace(-0.15, 1.15, 12), np.linspace(-0.15, 1.15, 12))
zz = 0.08 * xx + 0.08 * yy
ax.plot_surface(xx, yy, zz, alpha=0.12, color="gainsboro")

ax.set_title(
    "Schematic 5D orthogonal vector basis",
    fontsize=15)

ax.set_xlabel("Mass dimension (V1)", fontsize=10, labelpad=8)
ax.set_ylabel("Variance dimension (V2)", fontsize=10, labelpad=8)
ax.set_zlabel("Defect dimension (V3)", fontsize=10, labelpad=12)
ax.legend(loc="upper left", fontsize=9)
ax.view_init(elev=20, azim=45)
ax.set_xlim(-0.6, 1.1)
ax.set_ylim(-0.5, 1.1)
ax.set_zlim(-0.1, 1.1)


fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
_out_path = figures_dir / "step16_vector_space_5d.png"
plt.savefig(_out_path, dpi=300, bbox_inches=None)
print(f"  Saved figure: {_out_path}")
plt.close()