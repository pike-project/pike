import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import json
import subprocess
import argparse

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
        config = self.config

        level = config["level"]

        task_start = config["task_start"]
        task_end = config["task_end"]

        for task in range(task_start, task_end + 1):
            self.plot_task(level, task)

    def get_baseline_runtime(self, data, task):
        for v in data:
            if v["problem_id"] == task:
                return v["results"]["eval_results"]["runtime"]
        
        return None

    def get_baseline_runtimes(self, task):
        runtime_eager = self.get_baseline_runtime(self.baseline_eager, task)
        runtime_compile = self.get_baseline_runtime(self.baseline_compile, task)

        return (runtime_eager, runtime_compile)

    def plot_task(self, level, task):
        run_dir = self.run_dir
        
        phases_dir = run_dir / f"levels/level_{level}/task_{task}/phases"

        phase_nums = []

        for dirname in os.listdir(phases_dir):
            phase_nums.append(int(dirname.split("_")[1]))

        phase_count = np.max(np.array(phase_nums)) + 1

        print(f"Phase count: {phase_count}")

        phases = list(range(phase_count))
        best_runtimes = []

        all_solutions = []

        best_solutions_dir = Path.resolve(run_dir / "best_solutions")
        os.makedirs(best_solutions_dir, exist_ok=True)
        best_solution_path = best_solutions_dir / f"level_{level}_task_{task}.py"

        for phase in range(phase_count):
            phase_dir = phases_dir / f"phase_{phase}"
            solutions_dir = phase_dir / "solutions"

            phase_solutions = []

            for sol_dirname in os.listdir(solutions_dir):
                sol_dir = solutions_dir / sol_dirname
                data_path = sol_dir / "data.json"

                with open(data_path) as f:
                    data = json.load(f)
                    # print(f"Phase: {phase}, runtime: {data['runtime']}")
                    phase_solutions.append(data)

            all_solutions += phase_solutions
            all_solutions = sorted(all_solutions, key=lambda x: x["runtime"])
            phase_solutions = sorted(phase_solutions, key=lambda x: x["runtime"])

            best_sol = phase_solutions[0]
            best_sol_all = all_solutions[0]

            # best_runtime = best_sol["runtime"]
            best_runtime = best_sol_all["runtime"]

            best_runtimes.append(best_runtime)
            print(f"Phase {phase} best solution runtime: {best_runtime}")

        best_overall_sol = all_solutions[0]
        best_overall_runtime = best_overall_sol["runtime"]
        print(f"Best overall solution runtime: {best_overall_runtime}")

        with open(best_solution_path, "w") as f:
            f.write(best_overall_sol["code"])

        # df = pd.DataFrame(self.runtime_results)

        # f, ax = plt.subplots(1, figsize=(4, 3))

        # labels = df["label"]
        # values = df["runtime"]

        col = [
            "#2aadb6",
            "#ff6583",
            "#aa6fc5",
            "#ffa600",
        ]

        # colors = []

        # for _, row in df.iterrows():
        #     model_idx = row['model_idx']
        #     colors.append(col[model_idx])

        plt.figure(figsize=(4, 3))

        plt.title(f"Level {level} - Task {task} Improvement")
        plt.xlabel("Parallel Tree Search Phase")
        plt.ylabel('Runtime (ms)')

        plt.plot(phases, best_runtimes, linewidth=1)
        # plt.bar(labels, values, color=colors, linewidth=1, edgecolor='black')

        # plt.xticks(rotation=30, ha='right')
        # plt.grid(axis='y')
        # plt.gca().set_axisbelow(True)

        plt.xticks(phases)

        # plt.ylim(bottom=1.5, top=4)

        plt.subplots_adjust(left=0.2, bottom=0.18)

        baseline_runtime_eager, baseline_runtime_compile = self.get_baseline_runtimes(task)

        plt.axhline(y=baseline_runtime_eager, color=col[1], linestyle='--', linewidth=1, label='baseline eager')
        plt.axhline(y=baseline_runtime_compile, color=col[2], linestyle=':', linewidth=1, label='baseline compile')

        plt.legend(loc='upper right')

        fig_filename = f"level_{level}_task_{task}.pdf"

        figs_dir = Path.resolve(run_dir / "figs")
        os.makedirs(figs_dir, exist_ok=True)
        plt.savefig(figs_dir / fig_filename)

        figs_dir_2 = Path.resolve(curr_dir / "../../figs/improvement")
        os.makedirs(figs_dir_2, exist_ok=True)
        plt.savefig(figs_dir_2 / fig_filename)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str)
    args = parser.parse_args()

    # runs_dir = Path("/pscratch/sd/k/kir/llm/KernelBench-data/runs")
    # run_dir = runs_dir / run_id

    run_dir = Path(args.run_dir)

    plotter = ImprovementPlotter(run_dir)
    # plotter.backup()
    plotter.plot()

if __name__ == "__main__":
    main()
