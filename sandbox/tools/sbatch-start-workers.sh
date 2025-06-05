#!/bin/bash
#SBATCH -A m4141_g
#SBATCH -C gpu
#SBATCH -q interactive
#SBATCH -t 1:00:00
#SBATCH -N 1

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )/../.."

srun -n 1 -G 4 python3 $ROOT/sandbox/tools/start-worker-container.py
