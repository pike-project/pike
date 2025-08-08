# Evaluation Worker

The recommended way to run the eval worker is in a sandbox container like so:

```bash
python -u sandbox/tools/start_worker_container.py --engine docker --arch <Ampere/Hopper>
```

However, if you wish to run outside a sandbox container (not recommended), you can do so like this:

```bash
mkdir -p worker_io/input && mkdir -p worker_io/output && mkdir -p worker_io/scratch
python scripts/start_eval_worker.py --input_dir worker_io/input --output_dir worker_io/output --scratch_dir worker_io/scratch --arch <Ampere/Hopper>
```

## Design

TODO: describe the communication (explain why we use NFS to communicate, avoiding the need to make the network available to the container)
