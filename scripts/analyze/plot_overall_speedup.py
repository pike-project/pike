import os
from pathlib import Path
import matplotlib.pyplot as plt

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

def main():
    f, ax = plt.subplots(1, figsize=(3.25, 2.5))

    labels = [
        "No Ideas\nTop-3 Branch",
        "Init Ideas\nTop-4 Diverse"
    ]

    values = [
        1.086,
        1.16
    ]

    col = [
        "#2aadb6",
        "#ff6583",
        "#aa6fc5",
        "#ffa600",
    ]

    colors = []

    for idx in range(len(labels)):
        colors.append(col[idx])

    plt.title(f"Level 0 Speedups (compile)")
    plt.ylabel('Speedup')

    plt.bar(labels, values, color=colors, linewidth=1, edgecolor='black')

    plt.xticks(rotation=20, ha='right')
    plt.grid(axis='y')
    plt.gca().set_axisbelow(True)

    plt.subplots_adjust(left=0.25, bottom=0.3)

    figs_dir = Path.resolve(curr_dir / "../../figs/overall_speedup")

    os.makedirs(figs_dir, exist_ok=True)

    plt.savefig(figs_dir / "speedup1.pdf")

if __name__ == "__main__":
    main()
