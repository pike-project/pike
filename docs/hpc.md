# HPC Cluster Notes

## Perlmutter

Add the following to `~/.bashrc`:

```bash
alias alloc_gpu="salloc -N 1 -A m4141_g -t 3:00:00 -q interactive -C 'gpu&hbm40g'"
alias alloc_cpu="salloc -N 1 -A m4141 -t 3:00:00 -q interactive -C cpu"

module load conda
```

Migrate a Docker container on Perlmutter (for use in a job):

```bash
podman-hpc migrate <name>
```

## Lawrencium

Add the following to `~/.bashrc`:

```bash
module load miniconda3
```

How to use an H100 GPU node:

```bash
srun -A ac_binocular -t 0:30:00 --partition=es1 --qos=es_normal --gres=gpu:H100:1 --cpus-per-task=4 --pty /bin/bash
```
