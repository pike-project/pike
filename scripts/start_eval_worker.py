import time
import asyncio
from pathlib import Path
import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.util.disk_channel import DiskChannel

# The eval worker waits for new kernel tasks to arrive, then compiles and runs them

# it should make use of recv() and send(), possibly in some library since the sampler
# will also need to send the kernel tasks, then receive the result back from the eval worker

# recv should work by watching a directory for changes in a loop (may not be able to use watchdog due to NFS)
# and when a non-tmp file is found it processes it

# send should work by making a file.json.tmp, and then moving file.json.tmp to file.json only when the tmp
# file is confirmed to be written to disk

# the worker recv will default to "/input" (or you can pass in a path to an input dir)
# the worker send will default to "/output" (or you can pass in a path to an output dir)

# workers will need to have ids, then they will only get their dir exposed to them

# e.g.
#
# KernelBench-data/workers/0/input -> /input
# KernelBench-data/workers/0/output -> /output

# the worker is then responsible for load balancing the evaluations onto the GPUs that it has available to it

# we only technically need to evaluate the reference pytorch implementation once for each task, but the evaluation may
# not be the bottleneck of this anyway (the LLM sampling might be the bottleneck)
# -- it also seems that this is done separately with the current setup, as you can run scripts/generate_baseline_time.py
# to collect the baseline times for the current architecture

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/input")
    parser.add_argument("--output_dir", type=str, default="/output")
    args = parser.parse_args()

    tx_dir = Path(args.output_dir)
    rx_dir = Path(args.input_dir)

    disk_channel = DiskChannel(tx_dir, rx_dir)

    print("Eval worker running...")

    while True:
        msg = await disk_channel.recv()
        print(f"Got message: {msg}")

asyncio.run(main())
