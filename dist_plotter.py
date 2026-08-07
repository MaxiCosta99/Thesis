import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

"""# Dirichlet(alpha1, alpha2) == Beta(alpha1, alpha2)
Theta0, Theta1 = 2.3, 1.7   # <-- set your parameters here

fs = 'large'

x = np.linspace(0, 1, 100)
pdf = beta.pdf(x, Theta0, Theta1)


fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(x, pdf, lw=2)
ax.fill_between(x, pdf, alpha=0.2)
ax.set_xlabel(r"$\theta_0$", fontsize = 16)
ax.set_ylabel("Density", fontsize = 16)
ax.set_title(fr"$\mathrm{{Dir}}(\Theta_0={Theta0}, \Theta_1={Theta1})$", fontsize = 18)
ax.tick_params(labelsize = 13)

fig.tight_layout()

plt.show()"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.stats import dirichlet

# ---- Dirichlet parameters ----
Psi = np.array([5, 0.8, 1])   # Psi_1, Psi_2, Psi_3

# ---- Sample the simplex directly via barycentric coordinates ----
n = 400
x1 = np.linspace(0, 1, n)
x2 = np.linspace(0, 1, n)
X1, X2 = np.meshgrid(x1, x2)
X3 = 1 - X1 - X2
mask = X3 >= 0

X1v, X2v, X3v = X1[mask], X2[mask], X3[mask]

eps = 1e-3
keep = (X1v > eps) & (X2v > eps) & (X3v > eps)
X1v, X2v, X3v = X1v[keep], X2v[keep], X3v[keep]

pts = np.stack([X1v, X2v, X3v], axis=1)
density = dirichlet.pdf(pts.T, Psi)

# ---- Map barycentric -> equilateral triangle coordinates ----
v1 = np.array([0, 0])
v2 = np.array([1, 0])
v3 = np.array([0.5, np.sqrt(3) / 2])
centroid = (v1 + v2 + v3) / 3

def bary_to_xy(b1, b2, b3):
    x = b1 * v1[0] + b2 * v2[0] + b3 * v3[0]
    y = b1 * v1[1] + b2 * v2[1] + b3 * v3[1]
    return x, y

def outward_dir(x, y, length):
    dx, dy = x - centroid[0], y - centroid[1]
    norm = np.hypot(dx, dy)
    return dx / norm * length, dy / norm * length

Xp, Yp = bary_to_xy(X1v, X2v, X3v)

fig, ax = plt.subplots(figsize=(6, 5.5))

triang = mtri.Triangulation(Xp, Yp)
contour = ax.tricontourf(triang, density, levels=100, cmap="viridis")

triangle_x = [v1[0], v2[0], v3[0], v1[0]]
triangle_y = [v1[1], v2[1], v3[1], v1[1]]
ax.plot(triangle_x, triangle_y, color="black", linewidth=1.5, zorder=5)

tick_vals = np.arange(0.2, 1.0, 0.2)
tick_len = 0.03
label_len = 0.065

# Each edge fixes one barycentric coord at 0 and varies another by t.
# bottom (e1-e2): x3=0, varies x2=t      -> ticks are x2 values
# right  (e2-e3): x1=0, varies x3=t      -> ticks are x3 values
# left   (e3-e1): x2=0, varies x1=t      -> ticks are x1 values
edges = {
    "bottom": lambda t: (1 - t, t, 0),
    "right":  lambda t: (0, 1 - t, t),
    "left":   lambda t: (t, 0, 1 - t),
}
opposite_vertex = {"bottom": v3, "right": v1, "left": v2}

for edge_name, bary_fn in edges.items():
    for t in tick_vals:
        b1, b2, b3 = bary_fn(t)
        x_a, y_a = bary_to_xy(b1, b2, b3)

        tx, ty = outward_dir(x_a, y_a, tick_len)
        ax.plot([x_a, x_a + tx], [y_a, y_a + ty], color="black", lw=0.9, zorder=5)

        lx, ly = outward_dir(x_a, y_a, label_len)
        ax.text(x_a + lx, y_a + ly, f"{t:.1f}", ha="center", va="center", fontsize=7.5)

        ov = opposite_vertex[edge_name]
        ax.plot([x_a, ov[0]], [y_a, ov[1]], color="white", lw=0.4, alpha=0.35, zorder=2)

# Axis labels — corrected to match which coordinate each edge's ticks vary
ax.text(0.5, -0.20, r"$x_2$", ha="center", fontsize=11)          # bottom: x2
ax.text(0.80, 0.42, r"$x_3$", ha="center", fontsize=11, rotation=-60)  # right: x3
ax.text(0.20, 0.42, r"$x_1$", ha="center", fontsize=11, rotation=60)   # left: x1

ax.set_title(fr"Dirichlet density, $\Psi'=({Psi[0]}, {Psi[1]}, {Psi[2]})$", pad=12)
ax.set_aspect("equal")
ax.axis("off")
ax.set_xlim(-0.25, 1.25)
ax.set_ylim(-0.25, 0.98)
fig.colorbar(contour, ax=ax, shrink=0.75, label="density", pad=0.02)

plt.tight_layout(pad=0.5)
plt.savefig("dirichlet_simplex.png", dpi=300, bbox_inches="tight")
plt.show()