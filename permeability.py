
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 1. LOAD EXCEL
# ==========================================

df = pd.read_excel(
    "εκφώνηση_lab_1.xlsx",
    sheet_name="δεδομενα"
)

samples = {
    "1": (1, 2),
    "2": (5, 6),
    "3_1": (9, 10),
    "3_2": (13, 14)
}

colors = {
    "1": "tab:blue",
    "2": "tab:orange",
    "3_1": "tab:green",
    "3_2": "tab:red"
}

output_dir = Path("graphs")
output_dir.mkdir(exist_ok=True)


# ==========================================
# 2. NUMERICAL DERIVATIVE
# ==========================================

def calculate_derivative(H, B, window=31):

    # Local linear regression for each branch
    turn = np.argmin(H)

    branches = [
        (H[:turn + 1], B[:turn + 1]),
        (H[turn + 1:], B[turn + 1:])
    ]

    results = []

    for h, b in branches:

        order = np.argsort(h)
        h = h[order]
        b = b[order]

        # Average repeated H measurements
        h_unique, inverse, counts = np.unique(
            h, return_inverse=True, return_counts=True
        )

        b_unique = (
            np.bincount(inverse, weights=b) / counts
        )

        derivative = np.full(len(h_unique), np.nan)

        for i in range(len(h_unique)):

            start = max(0, i - window // 2)
            end = min(len(h_unique), i + window // 2 + 1)

            x = h_unique[start:end]
            y = b_unique[start:end]

            if len(x) < 3:
                continue

            dx = x - np.mean(x)
            denominator = np.sum(dx ** 2)

            if denominator > 0:
                derivative[i] = (
                    np.sum(dx * (y - np.mean(y)))
                    / denominator
                )

        results.append((h_unique, derivative))

    return results


# ==========================================
# 3. PLOT PERMEABILITY
# ==========================================

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(14, 6)
)

for sample, (h_col, b_col) in samples.items():

    data = df.iloc[:, [h_col, b_col]].dropna()

    H = data.iloc[:, 0].to_numpy(dtype=float) * 1000
    B = data.iloc[:, 1].to_numpy(dtype=float)

    color = colors[sample]
    label = f"Sample #{sample.replace('_', '.')}"

    # Split descending and ascending branches
    turn = np.argmin(H)

    branches = [
        (H[:turn + 1], B[:turn + 1]),
        (H[turn + 1:], B[turn + 1:])
    ]

    # --------------------------------------
    # mu(H) = B/H
    # --------------------------------------

    for j, (h, b) in enumerate(branches):

        mu = np.divide(
            b, h,
            out=np.full_like(b, np.nan),
            where=h != 0
        )

        # Avoid drawing across the singularity H=0
        crossings = np.where(h[:-1] * h[1:] < 0)[0]
        mu[crossings + 1] = np.nan

        ax1.plot(
            h / 1000, mu,
            color=color,
            label=label if j == 0 else None
        )

    # --------------------------------------
    # mu_diff(H) = dB/dH
    # --------------------------------------

    derivatives = calculate_derivative(H, B)

    for j, (h, mu_diff) in enumerate(derivatives):

        ax2.plot(
            h / 1000, mu_diff,
            color=color,
            label=label if j == 0 else None
        )


# ==========================================
# 4. GRAPH FORMATTING
# ==========================================

ax1.set_title("Magnetic Permeability μ(H)")
ax1.set_xlabel("H (kA/m)")
ax1.set_ylabel("μ (H/m)")
ax1.set_yscale("symlog", linthresh=2e-4)

ax2.set_title("Differential Permeability μdiff(H)")
ax2.set_xlabel("H (kA/m)")
ax2.set_ylabel("μdiff (H/m)")

for ax in (ax1, ax2):
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

plt.suptitle("Permeability Comparison - All Samples")
plt.tight_layout()

plt.savefig(
    output_dir / "permeability_all_samples.png",
    dpi=300
)

plt.close()

print("Permeability graphs saved successfully!")
