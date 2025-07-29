import os
from pathlib import Path
import matplotlib.pyplot as plt

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

def main():
    f, ax = plt.subplots(1, figsize=(3.25, 2.5))

    labels = [
        "Ours",
        "Ours (ideas)",
        "SI Lab Blog",
        "compile",

        "Ours",
        "METR",
        "compile",
    ]

    values = [
        1.67,
        1.79,
        1.52,
        1.54,

        2.29,
        2.46,
        1.4,
    ]

    colors = [
        "#2aadb6",
        "#2aadb6",
        "#ff6583",
        "#aa6fc5",

        "#2aadb6",
        "#ff6583",
        "#aa6fc5",
    ]

    x = list(range(len(labels)))
    bar_width = 0.8  # narrower bars for more spacing

    ax.bar(x, values, width=bar_width, color=colors, edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')

    # Dashed vertical divider between sections
    divider_index = 4
    ax.axvline(x=divider_index - 0.5, color='gray', linestyle='--', linewidth=1)

    # Section labels (not bold)
    ax.text(1.5, max(values) + 0.2, "Level 0", ha='center', fontsize=9)
    ax.text(5.25, max(values) + 0.2, "Level 3", ha='center', fontsize=9)

    ax.set_title("Speedups (eager)")
    ax.set_ylabel("Speedup")
    ax.grid(axis='y')
    ax.set_axisbelow(True)

    ax.set_ylim(0, 3)

    # Shift bars downward (increase bottom margin)
    plt.subplots_adjust(left=0.25, bottom=0.35)

    figs_dir = Path.resolve(curr_dir / "../../figs/overall_speedup_3_metr")
    os.makedirs(figs_dir, exist_ok=True)
    plt.savefig(figs_dir / "speedup1.pdf")

if __name__ == "__main__":
    main()
