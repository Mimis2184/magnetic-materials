import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Φόρτωση Excel
df = pd.read_excel(
    "εκφώνηση_lab_1.xlsx",
    sheet_name="δεδομενα"
)

# Μαγνητική διαπερατότητα κενού
mu0 = 4 * np.pi * 1e-7

# Αντιστοίχιση δειγμάτων με τις στήλες H και B
samples = {
    "1": (1, 2),
    "2": (5, 6),
    "3_1": (9, 10),
    "3_2": (13, 14)
}

# Φάκελος αποθήκευσης γραφημάτων
output_dir = Path("graphs")
output_dir.mkdir(exist_ok=True)

# Επεξεργασία κάθε δείγματος
for sample, (h_col, b_col) in samples.items():

    # Ανάγνωση των μετρήσεων
    data = df.iloc[:, [h_col, b_col]].dropna()
    H = data.iloc[:, 0].to_numpy(dtype=float)
    B = data.iloc[:, 1].to_numpy(dtype=float)

    # Υπολογισμός M
    H_Am = H * 1000
    M = B / mu0 - H_Am
    M_kAm = M / 1000

    # 1. Γραφική B(H)
    plt.figure(figsize=(8, 6))
    plt.plot(H, B)
    plt.xlabel("H (kA/m)")
    plt.ylabel("B (T)")
    plt.title(f"B(H) - Sample {sample}")
    plt.grid(True)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / f"BH_sample{sample}.png", dpi=300)
    plt.close()

    # 2. Γραφική M(H)
    plt.figure(figsize=(8, 6))
    plt.plot(H, M_kAm)
    plt.xlabel("H (kA/m)")
    plt.ylabel("M (kA/m)")
    plt.title(f"M(H) - Sample {sample}")
    plt.grid(True)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / f"MH_sample{sample}.png", dpi=300)
    plt.close()

    print(f"Completed sample {sample}")

print("All graphs saved successfully!")


# ===================================
# 3. COMPARISON PLOTS - ALL SAMPLES
# ===================================

colors = {
    "1": "tab:blue",
    "2": "tab:orange",
    "3_1": "tab:green",
    "3_2": "tab:red"
}

# Δημιουργία συγκριτικών διαγραμμάτων
for quantity in ["B", "M"]:

    plt.figure(figsize=(9, 6))

    for sample, (h_col, b_col) in samples.items():

        data = df.iloc[:, [h_col, b_col]].dropna()

        H = data.iloc[:, 0].to_numpy(dtype=float)
        B = data.iloc[:, 1].to_numpy(dtype=float)

        if quantity == "B":
            Y = B
            ylabel = "B (T)"
        else:
            H_Am = H * 1000
            M = B / mu0 - H_Am
            Y = M / 1000
            ylabel = "M (kA/m)"

        plt.plot(
            H, Y,
            color=colors[sample],
            label=f"Sample #{sample.replace('_', '.')}",
            linewidth=1.5
        )

    plt.xlabel("H (kA/m)")
    plt.ylabel(ylabel)
    plt.title(f"{quantity}(H) - All Samples")

    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)

    plt.grid(True)
    plt.legend(loc="upper left")
    plt.tight_layout()

    plt.savefig(
        output_dir / f"{quantity}H_all_samples.png",
        dpi=300
    )
    plt.close()

print("Comparison plots saved successfully!")

