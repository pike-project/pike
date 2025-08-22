# Profiling

Install profiling tools and use the eval script to profile a particular solution for a given task:

```bash
conda install -c nvidia nsight-compute

# (disable any profiling tools such as DCGMI currently using the GPU)
dcgmi profile --pause

mkdir -p ncu
ncu --set full --export ncu/minigpt.baseline-eager.ncu-rep python scripts/eval.py --level 3 --task 43 --code_path KernelBench/level3/43_MinGPTCausalAttention.py --profile
```

Note that issues such as the one below may come up, in which case you will need root access to enable profiling:

```
The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0. For instructions on enabling permissions and to get more information see https://developer.nvidia.com/ERR_NVGPUCTRPERM
```

## Torch Profiler

```python
model.eval()

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
    record_shapes=True,
    with_stack=True,
    profile_memory=True,
) as prof:
    with torch.no_grad():
        model(...)

# to export a chrome trace
prof.export_chrome_trace("trace.json")
```

To view with tensorboard, do the following:

```bash
pip install tensorboard torch-tb-profiler
tensorboard --logdir=./log
```

To view with Chrome tracing tools, go to:

```
chrome://tracing
```
