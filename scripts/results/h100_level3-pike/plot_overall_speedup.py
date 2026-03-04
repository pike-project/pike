import os
from pathlib import Path
import matplotlib.pyplot as plt

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

def main():
    # f, ax = plt.subplots(1, figsize=(6, 3.5))
    f, ax = plt.subplots(1, figsize=(6, 4.5))

    # EFA = Error Fixing Agent
    # IBA = Initial Brainstorming Agent

    labels = [
        "PIKE-B",
        "PIKE-B (cheap EFA)",
        "PIKE-B (no EFA)",
        "PIKE-B (no IBA)",

        "PIKE-O",
        "PIKE-O (no EFA)",
        "PIKE-O (mut)",

        "PIKE-O (mut,npar)",
        "PIKE-O (mut,npar,1isl)",
        "PIKE-O (mut,npar,1isl,EO)",
        "PIKE-O (mut,npar,1isl,EO,SL)",
        # "PIKE-O (mut2)",

        "METR",
        "torch.compile",
        "TensorRT",
    ]

    values = [
        2.8807218736524915,
        2.5937858635789635,
        1.980390794363127,
        2.6787584567497533,
        2.172204344053202,
        2.1521784752258135,
        2.0997138023406223,
        1.9895075738925703,
        2.749898113434464,
        2.383039666979026,
        2.8125091637254847,

        # 1.9070450822377587,

        1.40,
        1.64,
        1.41,
        # 1.34,
        # 1.49,
        # 1.34,
    ]

    # --- Optional checkpoint values ---
    # Use None to skip certain bars (last 3 skipped here)
    checkpoint_values = [
        2.306206146027268,
        2.506599235243787,
        1.7938174127305169,
        2.1370751650115234,
        1.8495042946288274,
        1.8274375981117181,
        1.8679863370873766,
        1.6284539655785533,
        2.1395040621647117,
        2.0098655600003186,
        2.330205924429939,

        # 1.7116044522358849,

        None,  # METR
        None,  # torch.compile
        None,  # TensorRT
    ]

    col = [
        "#2EBEC7",
        "#5BD0D4",
        "#5BD0D4",
        "#5BD0D4",

        "#ffa600",
        "#ffc559",
        "#ffc559",
        "#ffc559",
        "#ffc559",
        "#ffc559",
        "#ffc559",

        "#ff6583",
        "#c07bdf",
        "#CBA0DE",
    ]

    # plt.title("3-metr Speedups (Eager)")
    plt.ylabel('Speedup')

    bars = plt.bar(labels, values, color=col, linewidth=1, edgecolor='black')

    # --- Add numbers on top of bars ---
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha='center',
            va='bottom',
            fontsize=8,
            rotation=0,
        )

    # --- Add checkpoint lines ---
    for bar, checkpoint in zip(bars, checkpoint_values):
        if checkpoint is None:
            continue  # skip this bar
        x = bar.get_x()
        width = bar.get_width()

        # Draw solid gray checkpoint line
        ax.hlines(
            y=checkpoint,
            xmin=x + 0.02,
            xmax=x + width,
            color="black",
            linewidth=1.2,
            linestyle=":",
            alpha=0.6,
        )

        # Place checkpoint label just below the line
        ax.text(
            x + width / 2,
            checkpoint - 0.05,  # slightly below the line
            f"{checkpoint:.2f}",
            ha='center',
            va='top',
            fontsize=8,
            color='black',
            alpha=0.6,
        )

    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    plt.gca().set_axisbelow(True)

    # --- Layout adjustments ---
    plt.subplots_adjust(left=0.15, bottom=0.55, top=0.95)
    x_div = (len(values) - 3) - 0.5
    y_max = max(values)
    plt.axvline(x=x_div, linestyle="--", color="black", linewidth=0.8)
    plt.xlim(-0.5, len(labels) - 0.5)
    plt.ylim(0, y_max * 1.2)

    figs_dir = Path.resolve(curr_dir / "results/figs/overall_speedup")
    os.makedirs(figs_dir, exist_ok=True)
    plt.savefig(figs_dir / "overall_speedup.pdf")

if __name__ == "__main__":
    main()
