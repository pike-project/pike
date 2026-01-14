import subprocess
import re
from collections import defaultdict

# Nodes of interest
target_nodes = {"n0069.es2", "n0070.es2", "n0071.es2", "n0072.es2"}

def run_cmd(cmd):
    return subprocess.check_output(cmd, shell=True, text=True)

def get_jobs_on_nodes():
    jobs = defaultdict(list)  # node -> list of jobids
    squeue_out = run_cmd("squeue -h -o '%i %N'")
    for line in squeue_out.strip().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue  # skip malformed lines
        jobid, nodelist = parts[0], parts[1]
        # nodelist can contain multiple nodes, expand
        for node in nodelist.split(","):
            # Strip possible range suffixes, e.g. n0069[1-2] -> n0069
            node = re.sub(r"\[.*\]", "", node)
            if node in target_nodes:
                jobs[node].append(jobid)
    return jobs

def parse_job(jobid):
    job_out = run_cmd(f"scontrol show job {jobid}")
    gpus, cpus = 0, 0
    for token in job_out.split():
        if token.startswith("TresPerNode="):
            match = re.search(r"gpu:[A-Za-z0-9]+:(\d+)", token)
            if match:
                gpus += int(match.group(1))
        elif token.startswith("TresPerTask="):
            match = re.search(r"cpu=(\d+)", token)
            if match:
                cpus += int(match.group(1))
    return gpus, cpus

def main():
    jobs = get_jobs_on_nodes()
    node_usage = {node: {"gpus": 0, "cpus": 0} for node in target_nodes}
    
    for node, jobids in jobs.items():
        for jobid in jobids:
            g, c = parse_job(jobid)
            node_usage[node]["gpus"] += g
            node_usage[node]["cpus"] += c
    
    for node in target_nodes:
        gpus = node_usage[node]["gpus"]
        cpus = node_usage[node]["cpus"]
        print(f"{node}: GPUs={gpus}, CPUs={cpus}")

if __name__ == "__main__":
    main()
