
from pathlib import Path

import numpy as np
import pandas as pd

# ==========================================
# 1. SETTINGS
# ==========================================

ROOT = Path(__file__).resolve().parent
EXCEL = ROOT / "εκφώνηση_lab_1.xlsx"

RESULTS = ROOT / "results"

MU0 = 4 * np.pi * 1e-7
WINDOW = 31

samples = {
    "#1": (1, 2),
    "#2": (5, 6),
    "#3.1": (9, 10),
    "#3.2": (13, 14)
}


# ==========================================
# 2. LINEAR INTERPOLATION
# ==========================================

def value_at_zero(x, y):

    # Check whether x = 0 exists
    exact = np.flatnonzero(x == 0)

    if len(exact) > 0:
        return float(y[exact[0]])

    # Find consecutive points around zero
    crossings = np.flatnonzero(x[:-1] * x[1:] < 0)

    if len(crossings) != 1:
        raise ValueError(
            "Expected one zero crossing per branch"
        )

    i = crossings[0]

    # Linear interpolation
    result = y[i] - x[i] * (
        (y[i + 1] - y[i]) / (x[i + 1] - x[i])
    )

    return float(result)


# ==========================================
# 3. DIFFERENTIAL PERMEABILITY
# ==========================================

def calculate_derivative(H, B):

    # Sort measurements by H within one branch
    order = np.argsort(H)

    h = H[order]
    b = B[order]

    # Average repeated H values
    h_unique, inverse, counts = np.unique(
        h,
        return_inverse=True,
        return_counts=True
    )

    b_unique = (
        np.bincount(inverse, weights=b) / counts
    )

    derivative = np.full(len(h_unique), np.nan)

    # Local linear regression using 31 points
    for i in range(len(h_unique)):

        start = max(0, i - WINDOW // 2)
        end = min(
            len(h_unique),
            i + WINDOW // 2 + 1
        )

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

    return h_unique, derivative


# ==========================================
# 4. FIND MAXIMUM AND CORRESPONDING H
# ==========================================

def max_and_field(H, values):

    valid = np.isfinite(H) & np.isfinite(values)

    if not np.any(valid):
        raise ValueError("No valid values found")

    indices = np.flatnonzero(valid)
    i = indices[np.argmax(values[valid])]

    return float(values[i]), float(H[i])


# ==========================================
# 5. MAIN CALCULATIONS
# ==========================================

def main():

    df = pd.read_excel(
        EXCEL,
        sheet_name="δεδομενα"
    )

    summary_rows = []

    for name, (h_col, b_col) in samples.items():

        # Load H and B
        data = df.iloc[:, [h_col, b_col]].dropna()

        H = data.iloc[:, 0].to_numpy(dtype=float) * 1000
        B = data.iloc[:, 1].to_numpy(dtype=float)

        # Calculate magnetization M
        M = B / MU0 - H

        # ----------------------------------
        # A. Bmax AND Mmax
        # ----------------------------------

        B_max = float(np.max(B))
        M_max = float(np.max(M))

        # ----------------------------------
        # B. Mrem AND Hc
        # ----------------------------------

        # Split descending and ascending branches
        turn = int(np.argmin(H))

        branches = [
            (H[:turn + 1], B[:turn + 1]),
            (H[turn + 1:], B[turn + 1:])
        ]

        remanences = []
        coercivities = []

        differential_H = []
        differential_mu = []

        for h, b in branches:

            m = b / MU0 - h

            # Remanence: M when H = 0
            Mrem = value_at_zero(h, m)

            # Coercive field: H when B = 0
            Hc = value_at_zero(b, h)

            remanences.append(Mrem)
            coercivities.append(Hc)

            # Differential permeability
            hd, mud = calculate_derivative(h, b)

            differential_H.append(hd)
            differential_mu.append(mud)

        # Mean absolute values of the two branches
        Mrem_mean = np.mean(np.abs(remanences)) / 1000
        Hc_mean = np.mean(np.abs(coercivities))

        # ----------------------------------
        # C. PERMEABILITY mu = B/H
        # ----------------------------------

        mu = np.divide(
            B, H,
            out=np.full_like(B, np.nan),
            where=H != 0
        )

        mu_max, H_mu_max = max_and_field(H, mu)

        # ----------------------------------
        # D. SUSCEPTIBILITY chi = M/H
        # ----------------------------------

        chi = np.divide(
            M, H,
            out=np.full_like(M, np.nan),
            where=H != 0
        )

        chi_max, H_chi_max = max_and_field(H, chi)

        # ----------------------------------
        # E. DIFFERENTIAL QUANTITIES
        # ----------------------------------

        h_diff = np.concatenate(differential_H)
        mu_diff = np.concatenate(differential_mu)

        chi_diff = mu_diff / MU0 - 1

        mu_diff_max, H_mu_diff_max = max_and_field(
            h_diff, mu_diff
        )

        chi_diff_max, H_chi_diff_max = max_and_field(
            h_diff, chi_diff
        )

        # ----------------------------------
        # F. SAVE REQUIRED RESULTS
        # ----------------------------------

        summary_rows.append({
            "Sample": name,
            "Bmax (T)": B_max,
            "Mmax (kA/m)": M_max / 1000,
            "Mrem (kA/m)": Mrem_mean,
            "Hc (A/m)": Hc_mean,
            "mu_max (H/m)": mu_max,
            "H_at_mu_max (A/m)": H_mu_max,
            "mu_diff_max (H/m)": mu_diff_max,
            "H_at_mu_diff_max (A/m)": H_mu_diff_max,
            "chi_max": chi_max,
            "H_at_chi_max (A/m)": H_chi_max,
            "chi_diff_max": chi_diff_max,
            "H_at_chi_diff_max (A/m)": H_chi_diff_max
        })

    # ======================================
    # 6. EXPORT ONLY SUMMARY CSV
    # ======================================

    summary = pd.DataFrame(summary_rows)

    RESULTS.mkdir(exist_ok=True)

    output_file = RESULTS / "calculations_summary.csv"

    summary.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g"
    )

    # Delete old extra CSV if it exists
    old_file = RESULTS / "calculations_branches.csv"

    if old_file.exists():
        old_file.unlink()

    # Display results in terminal
    print("\n=== QUESTION B: RESULTS ===")

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.7g}"
        )
    )

    print("\nResults saved in:")
    print(output_file)


if __name__ == "__main__":
    main()
