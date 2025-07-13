import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import json
import subprocess
import argparse
import math
from scipy.stats import gmean

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

def get_baseline_runtime(path, task):
    with open(path) as f:
        data = json.load(f)
        for v in data:
            if v["problem_id"] == task:
                return v["results"]["eval_results"]["runtime"]

class ImprovementPlotter:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir

        with open(run_dir / "config.json") as f:
            self.config = json.load(f)
        
        self.level = self.config["level"]
        self.num_phases = self.config["num_phases"]

        with open(run_dir / "baseline_compile.json") as f:
            self.baseline_compile = json.load(f)
        
        with open(run_dir / "baseline_eager.json") as f:
            self.baseline_eager = json.load(f)

    def backup(self):
        run_dir = self.run_dir
        # last file/dir in path
        run_id = run_dir.name

        backup_dir = Path("/pscratch/sd/k/kir/llm/KernelBench-run-backups")
        backup_path = backup_dir / f"{run_id}.tar.gz"

        if not os.path.exists(backup_path):
            print("Backing up run...")

            cmd = ['tar', '-czvf', str(backup_path), str(run_dir)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            # print(result.stdout)
            print("Run backed up.")

    def plot(self):
        """
        Generates a single figure containing a grid of all task plots.
        """
        level = self.level
        num_phases = self.num_phases

        config = self.config
        task_start = config["task_start"]
        task_end = config["task_end"]

        tasks_to_plot = list(range(task_start, task_end + 1))
        num_plots = len(tasks_to_plot)

        if num_plots == 0:
            print("No tasks to plot.")
            return

        # Determine the grid size for the subplots
        cols = int(math.ceil(math.sqrt(num_plots)))
        rows = int(math.ceil(num_plots / cols))

        # Create a single large figure with a grid of subplots
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.5), squeeze=False)
        # Flatten the 2D array of axes for easy iteration
        axes = axes.flatten()

        print(f"Generating a {rows}x{cols} grid for {num_plots} tasks...")

        # TODO: need to take geomean of eager speedups and geomean of compile speedups
        # at each phase:
        # so build a numpy array and apply geomean along the correct dimension

        all_speedups_shape = (len(tasks_to_plot), num_phases)

        all_speedups_eager = np.zeros(all_speedups_shape)
        all_speedups_compile = np.zeros(all_speedups_shape)

        # Plot each task on its corresponding subplot
        for i, task in enumerate(tasks_to_plot):
            ax = axes[i]
            speedups_eager, speedups_compile = self.plot_task(level, task, ax)
            all_speedups_eager[i] = np.array(speedups_eager)
            all_speedups_compile[i] = np.array(speedups_compile)

        # print(all_speedups_eager.shape, all_speedups_compile.shape)

        geomean_eager = gmean(all_speedups_eager, axis=0)
        geomean_compile = gmean(all_speedups_compile, axis=0)

        # print(geomean_eager.shape, geomean_compile.shape)

        # Turn off any unused subplots in the grid
        for i in range(num_plots, len(axes)):
            axes[i].axis('off')

        # Add a title for the entire figure
        fig.suptitle(f"Improvement Plots for Level {level}", fontsize=20)
        
        # Adjust layout to prevent titles/labels from overlapping
        # The rect parameter makes space for the suptitle
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Define file paths for saving the combined figure
        fig_filename = f"level_{level}_all_tasks_improvement.pdf"

        self.save_fig(fig_filename)
        
        plt.close(fig)

        self.plot_speedup_geomeans(geomean_eager, geomean_compile)

    def plot_speedup_geomeans(self, geomean_eager, geomean_compile):
        level = self.level

        fig, ax = plt.subplots(figsize=(3, 2.5))

        plt.tight_layout(rect=[0.05, 0.05, 1, 0.96])

        phases = list(range(0, self.num_phases))

        ax.plot(phases, geomean_eager, label="eager")
        ax.plot(phases, geomean_compile, label="compile")

        ax.set_title(f"Level {level} Geomean Speedup")
        ax.set_xlabel("Parallel Tree Search Phase")
        ax.set_ylabel('Speedup')

        ax.legend(loc='upper left')

        ax.axhline(y=1, linestyle='--', linewidth=1.5, color='gray')

        filename = f"level_{level}_speedup_geomeans.pdf"

        self.save_fig(filename)

        plt.close(fig)

    def save_fig(self, filename):
        figs_dir = self.run_dir / "figs"
        os.makedirs(figs_dir, exist_ok=True)
        save_path1 = figs_dir / filename
        plt.savefig(save_path1)

        run_name = self.run_dir.name

        figs_dir_2 = curr_dir / f"../../figs/improvement/{run_name}"
        os.makedirs(figs_dir_2, exist_ok=True)
        save_path2 = figs_dir_2 / filename
        plt.savefig(save_path2)

    def get_baseline_runtime(self, data, task):
        for v in data:
            if v["problem_id"] == task:
                return v["results"]["eval_results"]["runtime"]
        
        return None

    def get_baseline_runtimes(self, task):
        runtime_eager = self.get_baseline_runtime(self.baseline_eager, task)
        runtime_compile = self.get_baseline_runtime(self.baseline_compile, task)
        return (runtime_eager, runtime_compile)

    # returns a pair of (speedups_eager, speedups_compile)
    def plot_task(self, level, task, ax):
        """
        Plots the improvement for a single task onto a given Axes object.
        """
        print(f"Plotting Level {level}, Task {task}...")
        run_dir = self.run_dir
        phases_dir = run_dir / f"levels/level_{level}/task_{task}/phases"

        if not os.path.exists(phases_dir):
            print(f"  Warning: Directory not found for task {task}. Skipping plot.")
            ax.text(0.5, 0.5, f"No data for Task {task}", ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            return

        phase_nums = [int(dirname.split("_")[1]) for dirname in os.listdir(phases_dir)]
        if not phase_nums:
            print(f"  Warning: No phases found for task {task}. Skipping plot.")
            ax.text(0.5, 0.5, f"No data for Task {task}", ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            return

        phase_count = np.max(np.array(phase_nums)) + 1
        phases = list(range(phase_count))
        best_runtimes = []
        all_solutions = []

        best_solutions_dir = self.run_dir / "best_solutions"
        os.makedirs(best_solutions_dir, exist_ok=True)
        best_solution_path = best_solutions_dir / f"level_{level}_task_{task}.py"

        for phase in range(phase_count):
            phase_dir = phases_dir / f"phase_{phase}"
            solutions_dir = phase_dir / "solutions"
            phase_solutions = []

            for sol_dirname in os.listdir(solutions_dir):
                data_path = solutions_dir / sol_dirname / "data.json"
                with open(data_path) as f:
                    data = json.load(f)
                    phase_solutions.append(data)
            
            if not phase_solutions:
                best_runtimes.append(np.nan) # Use NaN for missing data
                continue

            all_solutions.extend(phase_solutions)
            all_solutions.sort(key=lambda x: x["runtime"])
            best_runtime_so_far = all_solutions[0]["runtime"]
            best_runtimes.append(best_runtime_so_far)

        if not all_solutions:
            print(f"  Warning: No solutions found for task {task}. Skipping plot.")
            ax.text(0.5, 0.5, f"No solution data for Task {task}", ha='center', va='center')
            return

        best_overall_sol = all_solutions[0]
        with open(best_solution_path, "w") as f:
            f.write(best_overall_sol["code"])

        col = ["#2aadb6", "#ff6583", "#aa6fc5", "#ffa600"]

        ax.set_title(f"Level {level} - Task {task} Improvement")
        ax.set_xlabel("Parallel Tree Search Phase")
        ax.set_ylabel('Runtime (ms)')

        ax.plot(phases, best_runtimes, linewidth=1.5, marker='o', markersize=4, label='Best Found')
        ax.set_xticks(phases)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

        baseline_runtime_eager, baseline_runtime_compile = self.get_baseline_runtimes(task)
        if baseline_runtime_eager is not None:
            ax.axhline(y=baseline_runtime_eager, color=col[1], linestyle='--', linewidth=1.5, label='Baseline Eager')
        if baseline_runtime_compile is not None:
            ax.axhline(y=baseline_runtime_compile, color=col[2], linestyle=':', linewidth=1.5, label='Baseline Compile')

        ax.legend(loc='upper right')

        speedups_eager = []
        speedups_compile = []

        for runtime in best_runtimes:
            speedups_eager.append(baseline_runtime_eager / runtime)
            speedups_compile.append(baseline_runtime_compile / runtime)
        
        return speedups_eager, speedups_compile

def main():
    parser = argparse.ArgumentParser(description="Plot improvement across multiple tasks from a run directory.")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to the specific run directory.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"Error: Run directory not found at '{run_dir}'")
        return

    plotter = ImprovementPlotter(run_dir)
    # plotter.backup()
    plotter.plot()

if __name__ == "__main__":
    # Example usage from command line:
    # python your_script_name.py --run_dir /path/to/your/run/directory
    main()
