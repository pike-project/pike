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
        # ------ TODO: dashed divide needed here in the bar graph -------
        "Ours (2)",
        "METR",
        "compile (2)",
        # "Ours: No Ideas\nTop-3 Branch",
        # "Ours: Init Ideas\nTop-4 Branch",
        # "SI Lab Blog\n(not A100-tuned)"
    ]

    values = [
        # 1.086,
        # 1.16,
        # 0.984

        # 2.84,
        # 1.77,

        1.67, # (Level 0 eager - ours)
        1.79, # (Level 0 eager - ours, ideas)
        1.52, # (Level 0 eager - blog post)
        1.54, # compile

        2.29, # (Level 3 eager - ours)
        2.46, # (Level 3 eager - metr)
        1.4, # compile
    ]

    col = [
        "#2aadb6",
        "#2aadb6",
        "#ff6583",
        "#aa6fc5",

        "#2aadb6",
        "#ff6583",
        "#aa6fc5",
    ]

    # "#aa6fc5",
    # "#ffa600",

    colors = []

    for idx in range(len(labels)):
        colors.append(col[idx])

    plt.title(f"Speedups (eager)")
    plt.ylabel('Speedup')

    plt.bar(labels, values, color=colors, linewidth=1, edgecolor='black')

    plt.xticks(rotation=25, ha='right')
    plt.grid(axis='y')
    plt.gca().set_axisbelow(True)

    plt.subplots_adjust(left=0.25, bottom=0.25)

    figs_dir = Path.resolve(curr_dir / "../../figs/overall_speedup_3_metr")

    os.makedirs(figs_dir, exist_ok=True)

    plt.savefig(figs_dir / "speedup1.pdf")

if __name__ == "__main__":
    main()
