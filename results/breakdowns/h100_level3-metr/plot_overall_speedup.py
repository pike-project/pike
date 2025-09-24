import os
from pathlib import Path
import matplotlib.pyplot as plt

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

def main():
    f, ax = plt.subplots(1, figsize=(4.5, 3.5))

    labels = [
        "prev. agents",
        "OE, pop 25, agents",
        "OE, pop 25, no agents",
        "OE, pop 25, no agents*",
        "OE, pop 10, no agents",
        "METR",
        "torch.compile",
        "TensorRT",
    ]

    values = [
        2.03,
        1.94,
        1.83,
        1.8,
        1.69,
        1.34,
        1.49,
        1.34,
    ]

    col = [
        "#2aadb6",
        "#8bc346",
        "#b2e376",
        "#b2e376",
        "#b2e376",

        "#ffa600",
        "#ff6583",
        "#c07bdf",
    ]

    plt.title("3-metr Speedups (Eager)")
    plt.ylabel('Speedup')

    bars = plt.bar(labels, values, color=col, linewidth=1, edgecolor='black')

    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    plt.gca().set_axisbelow(True)

    # Adjust layout
    plt.subplots_adjust(left=0.25, bottom=0.4)

    # --- Add dividing line after 5th bar ---
    x_div = 4.5  # between 4th and 5th index
    y_max = max(values)
    plt.axvline(x=x_div, linestyle="--", color="black", linewidth=0.8)

    # --- Add "Ours" label just above the left bars ---
    plt.text(
        x=2,  # centered over first 5 bars
        y=y_max + 0.05 * y_max,  # slightly above bars
        s="Ours",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    # --- Give more horizontal breathing room ---
    plt.xlim(-0.5, len(labels) - 0.5)

    # --- Give more vertical space for text ---
    plt.ylim(0, y_max * 1.2)

    figs_dir = Path.resolve(curr_dir / "results/figs/overall_speedup_3_metr")
    os.makedirs(figs_dir, exist_ok=True)
    plt.savefig(figs_dir / "speedup1.pdf")

if __name__ == "__main__":
    main()
