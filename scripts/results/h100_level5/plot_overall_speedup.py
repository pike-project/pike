import os
from pathlib import Path
import matplotlib.pyplot as plt

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

def main():
    f, ax = plt.subplots(1, figsize=(4.5, 3.5))

    # EFA = Error Fixing Agent
    # IBA = Initial Brainstorming Agent

    labels = [
        "PIKE-B",
        "PIKE-O",
        "METR",
        "torch.compile",
        "TensorRT",
    ]

    values = [
        1.25,
        2.25,
        1.46,
        1.29,
        1.25,
    ]

    col = [
        "#2EBEC7",
        "#8bc346",
        # "#ffa600",
        "#ff6583",
        "#c07bdf",
        "#CBA0DE",
    ]

    plt.title("Level 5 Speedups (Eager)")
    plt.ylabel('Speedup')

    bars = plt.bar(labels, values, color=col, linewidth=1, edgecolor='black')

    # Add numbers on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha='center',
            va='bottom',
            fontsize=10,
            rotation=0,
        )

    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    plt.gca().set_axisbelow(True)

    # Adjust layout
    plt.subplots_adjust(left=0.15, bottom=0.27)

    # --- Add dividing line after 5th bar ---
    # x_div = 0.5  # between 4th and 5th index
    y_max = max(values)
    # plt.axvline(x=x_div, linestyle="--", color="black", linewidth=0.8)

    # --- Add "Ours" label just above the left bars ---
    # plt.text(
    #     x=2.5,  # centered over first 5 bars
    #     y=y_max + 0.05 * y_max,  # slightly above bars
    #     s="Ours",
    #     ha="center",
    #     va="bottom",
    #     fontsize=10,
    # )

    # --- Give more horizontal breathing room ---
    plt.xlim(-0.5, len(labels) - 0.5)

    # --- Give more vertical space for text ---
    plt.ylim(0, y_max * 1.2)

    x_div = 1.5
    plt.axvline(x=x_div, linestyle="--", color="black", linewidth=0.8)

    figs_dir = Path.resolve(curr_dir / "results/figs/overall_speedup")
    os.makedirs(figs_dir, exist_ok=True)
    plt.savefig(figs_dir / "overall_speedup.pdf")

if __name__ == "__main__":
    main()
