"""Magnetic materials exercise: comparative chi(H) and chi_diff(H).

Run in WSL from the same folder as 'εκφώνηση_lab_1.xlsx':
    python3 "susceptibility.py"

Requires: numpy, pandas, matplotlib, openpyxl.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save PNGs without requiring a graphical window in WSL.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
EXCEL = ROOT / "εκφώνηση_lab_1.xlsx"
OUTPUT = ROOT / "graphs"
OUTPUT.mkdir(exist_ok=True)
MU0 = 4 * np.pi * 1e-7

SAMPLES = {
    "#1": (1, 2, "tab:blue"),
    "#2": (5, 6, "tab:orange"),
    "#3.1": (9, 10, "tab:green"),
    "#3.2": (13, 14, "tab:red"),
}


def branch_derivatives(h_am: np.ndarray, b_t: np.ndarray, window: int = 31):
    """Calculate local dB/dH using the two hysteresis branches separately."""
    turn = int(np.argmin(h_am))
    branches = ((h_am[:turn + 1], b_t[:turn + 1]),
                (h_am[turn + 1:], b_t[turn + 1:]))
    results = []

    for h, b in branches:
        order = np.argsort(h)
        h_sorted, b_sorted = h[order], b[order]
        h_unique, inv, counts = np.unique(
            h_sorted, return_inverse=True, return_counts=True
        )
        b_average = np.bincount(inv, weights=b_sorted) / counts
        slopes = np.full(h_unique.shape, np.nan, dtype=float)

        for i in range(len(h_unique)):
            lo = max(0, i - window // 2)
            hi = min(len(h_unique), i + window // 2 + 1)
            x, y = h_unique[lo:hi], b_average[lo:hi]
            xc = x - x.mean()
            denom = float(xc @ xc)
            if len(x) >= 3 and denom > 0:
                slopes[i] = float((xc @ (y - y.mean())) / denom)

        results.append((h_unique, slopes))

    return results


def plot_ratio_branches(ax, h_am, b_t, color, name):
    """Plot chi=M/H = B/(mu0*H)-1, breaking at H=0."""
    turn = int(np.argmin(h_am))
    for j, sl in enumerate((slice(None, turn + 1), slice(turn + 1, None))):
        h, b = h_am[sl], b_t[sl]
        chi = np.divide(b, MU0 * h, out=np.full_like(b, np.nan), where=h != 0) - 1
        crosses_zero = np.flatnonzero(h[:-1] * h[1:] < 0)
        chi[crosses_zero + 1] = np.nan
        ax.plot(h / 1000, chi, color=color, lw=1.4,
                label=f"Sample {name}" if j == 0 else None)


def main():
    df = pd.read_excel(EXCEL, sheet_name="δεδομενα")
    fig, (ax_chi, ax_diff) = plt.subplots(1, 2, figsize=(14, 5.8))

    for name, (h_col, b_col, color) in SAMPLES.items():
        data = df.iloc[:, [h_col, b_col]].dropna().to_numpy(dtype=float)
        h_am = data[:, 0] * 1000.0  # kA/m -> A/m
        b_t = data[:, 1]

        # chi = M/H = B/(mu0*H) - 1 (dimensionless)
        plot_ratio_branches(ax_chi, h_am, b_t, color, name)

        # chi_diff = dM/dH = (dB/dH)/mu0 - 1 (dimensionless)
        for j, (h_unique, db_dh) in enumerate(branch_derivatives(h_am, b_t)):
            ax_diff.plot(h_unique / 1000, db_dh / MU0 - 1,
                         color=color, lw=1.6,
                         label=f"Sample {name}" if j == 0 else None)

    ax_chi.set_title("Magnetic susceptibility  χ(H) = M/H")
    ax_chi.set_ylabel("χ (dimensionless)")
    ax_chi.set_yscale("symlog", linthresh=200)
    ax_chi.text(0.02, 0.02, "Signed logarithmic vertical scale",
                transform=ax_chi.transAxes, fontsize=8, color="dimgray")

    ax_diff.set_title("Differential susceptibility  χdiff(H) = dM/dH")
    ax_diff.set_ylabel("χdiff (dimensionless)")

    for ax in (ax_chi, ax_diff):
        ax.set_xlabel("H (kA/m)")
        ax.axhline(0, color="black", lw=0.65)
        ax.axvline(0, color="black", lw=0.65)
        ax.grid(True, alpha=0.28)
        ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("Magnetic Materials — Susceptibility Comparison", fontsize=14)
    fig.tight_layout()
    path = OUTPUT / "susceptibility_all_samples.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    print("Note: χ near H=0 is singular; χdiff uses local fits (31 points per branch).")


if __name__ == "__main__":
    main()
