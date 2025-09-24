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

def get_task_label(level, task):
    task_labels = []

    if level == 0:
        task_labels = [
            "Conv2D",
            "Conv2D-ReLU-MaxPool",
            "LayerNorm",
            "MatMul",
            "Softmax"
        ]

    # 1-indexed task numbers
    idx = task - 1
    if idx < 0:
        return None

    if idx >= len(task_labels):
        return None
    
    return task_labels[idx]

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
        
        # self.cuda_streams_solutions = [6, 8, 9, 14, 22, 23, 25, 40, 41, 48]
        # self.cuda_streams_solutions = [6, 14]
        self.cuda_streams_solutions = []

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
        If there are fewer plots on the bottom row, they will be centered
        while maintaining the same size as plots in the rows above.
        """
        level = self.level
        num_phases = self.num_phases

        config = self.config
        task_start = config["task_start"]
        task_end = config["task_end"]

        level_dir = self.run_dir / f"levels/level_{level}"

        tasks_to_plot = []

        for task_dirname in os.listdir(level_dir):
            task = int(task_dirname.split("_")[1])
            # TODO: figure out why torch.compile is broken here
            if level == "3-metr":
                if task == 33:
                    continue

            tasks_to_plot.append(task)

        tasks_to_plot = sorted(tasks_to_plot)

        # tasks_to_plot = list(range(task_start, task_end + 1))
        num_plots = len(tasks_to_plot)

        if num_plots == 0:
            print("No tasks to plot.")
            return

        # Determine the grid size for the subplots
        cols = int(math.ceil(math.sqrt(num_plots)))
        rows = int(math.ceil(num_plots / cols))

        # Create a figure and a fine-grained GridSpec. Each plot will span 2 columns.
        fig = plt.figure(figsize=(cols * 5, rows * 4.5))
        gs = fig.add_gridspec(rows, cols * 2, wspace=0.6, hspace=0.45)

        axes_list = []
        
        # Calculate the number of plots in the last row
        num_in_last_row = num_plots % cols
        if num_in_last_row == 0 and num_plots > 0:
            num_in_last_row = cols

        print(f"Generating a {rows}x{cols} grid for {num_plots} tasks (with centering)...")

        for i in range(num_plots):
            current_row = i // cols
            col_in_row = i % cols

            offset = 0
            # If we are in the last row, calculate the offset to center the plots
            if current_row == rows - 1:
                total_cells = cols * 2
                plots_width_in_cells = num_in_last_row * 2
                empty_space_in_cells = total_cells - plots_width_in_cells
                offset = empty_space_in_cells // 2

            # Each plot spans 2 columns in the fine-grained grid
            start_col = offset + col_in_row * 2
            ax = fig.add_subplot(gs[current_row, start_col:start_col + 2])
            axes_list.append(ax)

        axes = axes_list
        
        all_speedups_shape = (len(tasks_to_plot), num_phases)
        all_speedups_eager = np.zeros(all_speedups_shape)
        all_speedups_compile = np.zeros(all_speedups_shape)
        max_phases_completed = 0

        # Plot each task on its corresponding subplot
        for i, task in enumerate(tasks_to_plot):
            ax = axes[i]
            plot_data = self.plot_task(level, task, ax)
            
            # Skip if plot_task returned no data
            if plot_data is None:
                continue
            
            speedups_eager, speedups_compile = plot_data
            all_speedups_eager[i, :len(speedups_eager)] = np.array(speedups_eager)
            all_speedups_compile[i, :len(speedups_compile)] = np.array(speedups_compile)

            max_phases_completed = max(max_phases_completed, len(speedups_eager))

        # Trim arrays to the actual max number of phases completed across all tasks
        all_speedups_eager = all_speedups_eager[:, :max_phases_completed]
        all_speedups_compile = all_speedups_compile[:, :max_phases_completed]

        all_speedups_eager_final = all_speedups_eager[:, -1].tolist()
        all_speedups_compile_final = all_speedups_compile[:, -1].tolist()
        self.handle_all_speedups_final(tasks_to_plot, all_speedups_eager_final, all_speedups_compile_final)

        # Calculate geomean, ignoring NaNs
        geomean_eager = gmean(all_speedups_eager[~np.isnan(all_speedups_eager).any(axis=1)], axis=0)
        geomean_compile = gmean(all_speedups_compile[~np.isnan(all_speedups_compile).any(axis=1)], axis=0)

        # Add a title for the entire figure
        fig.suptitle(f"Improvement Plots for Level {level}", fontsize=20)
        
        # Adjust layout to prevent titles/labels from overlapping
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        # Define file paths for saving the combined figure
        fig_filename = f"level_{level}_all_tasks_improvement.pdf"

        self.save_fig(fig, fig_filename)
        
        plt.close(fig)

        if geomean_eager.size > 0 and geomean_compile.size > 0:
            eager_final = geomean_eager[-1]
            compile_final = geomean_compile[-1]

            print(f"Final eager speedup: {eager_final}, final compile speedup: {compile_final}")

            self.plot_speedup_geomeans(geomean_eager, geomean_compile)

    def plot_speedup_geomeans(self, geomean_eager, geomean_compile):
        level = self.level

        fig, ax = plt.subplots(figsize=(3, 2.5))

        plt.tight_layout(rect=[0.05, 0.05, 1, 0.96])

        phases = list(range(0, geomean_eager.shape[0]))

        ax.plot(phases, geomean_eager, label="eager")
        ax.plot(phases, geomean_compile, label="compile")

        ax.set_title(f"Level {level} gmean Speedup")
        ax.set_xlabel("Parallel Tree Search Phase")
        ax.set_ylabel('Speedup')

        ax.legend(loc='upper left')

        ax.axhline(y=1, linestyle='--', linewidth=1.5, color='gray')

        filename = f"level_{level}_speedup_geomeans.pdf"

        self.save_fig(fig, filename)

        plt.close(fig)

    def handle_all_speedups_final(self, tasks_to_plot, all_speedups_eager_final, all_speedups_compile_final):
        print(f"All speedups eager: {all_speedups_eager_final}")

        try:
            with open(self.run_dir / "comp_runtimes.json") as f:
                comp_runtimes = json.load(f)
        except Exception as e:
            return
    
        assert len(all_speedups_eager_final) == len(tasks_to_plot), "All speedups eager final length should be same as tasks_to_plot length"
        assert len(all_speedups_compile_final) == len(tasks_to_plot), "All speedups compile final length should be same as tasks_to_plot length"

        included_tasks = []
        v_rels = []
        compile_rels = []
        eager_rels = []

        for idx, task in enumerate(tasks_to_plot):
            eager_runtime, compile_runtime = self.get_baseline_runtimes(task)
            v_runtime = self.get_baseline_runtime(comp_runtimes, task)

            if v_runtime is None:
                continue
            
            our_speedup = all_speedups_eager_final[idx]
            v_speedup = eager_runtime / v_runtime
            compile_speedup = eager_runtime / compile_runtime

            v_rel = our_speedup / v_speedup
            compile_rel = our_speedup / compile_speedup

            v_rels.append(v_rel)
            compile_rels.append(compile_rel)
            eager_rels.append(our_speedup)
            included_tasks.append(task)

        fig, ax = plt.subplots(figsize=(13, 2.5))

        ax.plot(included_tasks, v_rels, label="metr", marker='o', markersize=4)
        ax.plot(included_tasks, compile_rels, label="compile", marker='o', markersize=4)
        # ax.plot(included_tasks, eager_rels, label="eager", marker='o', markersize=4)

        plt.title("Level 3 (filtered) Runtimes Relative to METR/torch.compile (H100)")

        plt.xticks(range(1, 51))
        plt.grid(True, axis='x', which='both', linestyle='--', linewidth=0.5, alpha=0.3)
        plt.grid(True, axis='y', which='both', linestyle='--', linewidth=0.5, alpha=0.3)

        plt.ylabel("Task Number")
        plt.ylabel("Relative Runtime")

        plt.yscale('log')
        plt.axhline(y=1, color='gray', linestyle='--', linewidth=1)

        ax.legend(loc='upper right')

        filename = "individual_breakdown.pdf"
        self.save_fig(fig, filename)

    def save_fig(self, fig_object, filename):
        figs_dir = self.run_dir / "figs"
        os.makedirs(figs_dir, exist_ok=True)
        save_path1 = figs_dir / filename
        fig_object.savefig(save_path1)

        run_name = self.run_dir.name

        figs_dir_2 = curr_dir / f"../../results/figs/improvement/{run_name}"
        os.makedirs(figs_dir_2, exist_ok=True)
        save_path2 = figs_dir_2 / filename
        fig_object.savefig(save_path2)

    def get_baseline_runtime(self, data, task):
        for v in data:
            if v["problem_id"] == task:
                try:
                    return v["results"]["eval_results"]["runtime"]
                except Exception as e:
                    return None
        
        return None

    def get_baseline_runtimes(self, task):
        runtime_eager = self.get_baseline_runtime(self.baseline_eager, task)
        runtime_compile = self.get_baseline_runtime(self.baseline_compile, task)
        return (runtime_eager, runtime_compile)

    def plot_task(self, level, task, ax):
        """
        Plots the improvement for a single task onto a given Axes object.
        Returns None if no data is found, otherwise returns speedups.
        """
        run_dir = self.run_dir
        phases_dir = run_dir / f"levels/level_{level}/task_{task}/phases"

        if not os.path.exists(phases_dir):
            print(f"  Warning: Directory not found for task {task}. Skipping plot.")
            ax.text(0.5, 0.5, f"No data for Task {task}", ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            return None

        phase_dirs = [d for d in os.listdir(phases_dir) if d.startswith("phase_")]
        if not phase_dirs:
            print(f"  Warning: No phases found for task {task}. Skipping plot.")
            ax.text(0.5, 0.5, f"No data for Task {task}", ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            return None
        
        phases = list(range(self.num_phases))
        best_runtimes = []
        all_solutions = []

        best_solutions_dir = self.run_dir / "best_solutions"
        os.makedirs(best_solutions_dir, exist_ok=True)
        best_solution_path = best_solutions_dir / f"level_{level}_task_{task}.py"

        for phase in range(self.num_phases):
            phase_dir = phases_dir / f"phase_{phase}"
            solutions_dir = phase_dir / "solutions"
            phase_solutions = []

            if os.path.isdir(solutions_dir):
                for sol_dirname in os.listdir(solutions_dir):
                    data_path = solutions_dir / sol_dirname / "data.json"
                    if os.path.exists(data_path):
                        with open(data_path) as f:
                            data = json.load(f)
                            phase_solutions.append(data)
                
            # if len(phase_solutions) == 0:
            #     best_runtimes.append(np.nan)
            #     continue

            all_solutions.extend(phase_solutions)
            all_solutions.sort(key=lambda x: x["runtime"])
            best_runtime_so_far = all_solutions[0]["runtime"]
            best_runtimes.append(best_runtime_so_far)

        if not all_solutions:
            print(f"  Warning: No solutions found for task {task}. Skipping plot.")
            ax.text(0.5, 0.5, f"No solution data for Task {task}", ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            return None

        best_overall_sol = all_solutions[0]
        with open(best_solution_path, "w") as f:
            f.write(best_overall_sol["code"])

        col = ["#2aadb6", "#ff6583", "#aa6fc5", "#ffa600", "#8bc346"]
        
        task_label_str = get_task_label(level, task)
        title_task_label = f" ({task_label_str})" if task_label_str else ""

        ax.set_title(f"Level {level} - Task {task}{title_task_label}")
        ax.set_xlabel("Parallel Tree Search Phase")
        ax.set_ylabel('Runtime (ms)')

        ax.plot(phases, best_runtimes, linewidth=3, marker='o', markersize=6, label='Best Found')
        ax.set_xticks(phases)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

        baseline_runtime_eager, baseline_runtime_compile = self.get_baseline_runtimes(task)
        if baseline_runtime_eager is not None:
            ax.axhline(y=baseline_runtime_eager, color=col[4], linestyle='--', linewidth=3, label='Baseline Eager')
        if baseline_runtime_compile is not None:
            ax.axhline(y=baseline_runtime_compile, color=col[1], linestyle=':', linewidth=3, label='Baseline Compile')

        ax.legend(loc='upper right')

        speedups_eager = []
        speedups_compile = []

        for runtime in best_runtimes:
            if baseline_runtime_eager is not None and not np.isnan(runtime) and runtime > 0:
                speedup = baseline_runtime_eager / runtime
                if speedup < 1 or task in self.cuda_streams_solutions:
                    # print(f"Task {task} speedup is < 1, using baseline as best solution")
                    speedup = 1

                speedups_eager.append(speedup)
            else:
                speedups_eager.append(np.nan)
            
            if baseline_runtime_compile is not None and not np.isnan(runtime) and runtime > 0:
                speedup = baseline_runtime_compile / runtime
                if speedup < 1 or task in self.cuda_streams_solutions:
                    # print(f"Task {task} speedup is < 1, using baseline as best solution")
                    speedup = 1

                speedups_compile.append(speedup)
            else:
                speedups_compile.append(np.nan)

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
