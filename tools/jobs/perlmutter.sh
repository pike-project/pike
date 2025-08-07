#!/bin/bash

srun -N 1 -A m4141_g -t 24:00:00 -q regular -C 'gpu&hbm40g' --pty python -u sandbox/tools/start_worker_container.py --engine podman-hpc --arch Ampere --pull_image
