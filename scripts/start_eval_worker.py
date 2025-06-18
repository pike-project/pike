import time
import asyncio
import aiofiles
from pathlib import Path
import sys
import os
import argparse
import uuid

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.util.disk_channel import DiskChannel

curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))

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

class EvalWorker:
    def __init__(self, tx_dir: Path, rx_dir: Path, scratch_dir: Path):
        self.tx_dir = tx_dir
        self.rx_dir = rx_dir
        self.scratch_dir = scratch_dir

        self.code_dir = self.scratch_dir / "code"
        os.makedirs(self.code_dir, exist_ok=True)

        self.eval_output_dir = self.scratch_dir / "eval_output"
        os.makedirs(self.eval_output_dir, exist_ok=True)

        self.disk_channel = DiskChannel(tx_dir, rx_dir)

        self.eval_script_path = curr_dir / "eval.py"
    
    async def run(self):
        print("Eval worker running...")

        while True:
            msg = await self.disk_channel.recv()
            # print(f"Got message: {msg}")

            level = msg["level"]
            task = msg["task"]
            code_str = msg["code"]

            # 1. write the LLM-generated code to scratch dir with a unique name
            file_id = str(uuid.uuid4())
            code_path = self.code_dir / f"task_{level}_{task}_{file_id}.py"
            async with aiofiles.open(code_path, 'w', encoding='utf-8') as f:
                await f.write(code_str)
            
            print(f"Wrote: {code_path}")

            eval_output_path = self.eval_output_dir / f"task_{level}_{task}_{file_id}.json"

            # 2. invoke scripts/eval.py with the level, task, and path to the LLM-generated code
            #    (do not wait for this to finish, keep listening for tasks to start)

            cmd = ["python", str(self.eval_script_path), "--level", str(level), "--task", str(task), "--code_path", str(code_path), "--output_path", str(eval_output_path)]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            print(f"[stdout]\n{stdout.decode()}")
            if stderr:
                print(f"[stderr]\n{stderr.decode()}")

            # 3. read results back, then send them out to the disk_channel 

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/input")
    parser.add_argument("--output_dir", type=str, default="/output")
    parser.add_argument("--scratch_dir", type=str, default="/scratch")
    args = parser.parse_args()

    tx_dir = Path(args.output_dir)
    rx_dir = Path(args.input_dir)
    scratch_dir = Path(args.scratch_dir)

    eval_worker = EvalWorker(tx_dir, rx_dir, scratch_dir)
    await eval_worker.run()

asyncio.run(main())
