import time
import asyncio
import aiofiles
from pathlib import Path
import sys
import os
import argparse
import json

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


# this function ensures that the results coming back from the LLM evaluation are sanitized safely
def assert_type_and_unpack(untrusted_dict: dict, src_key: str, expected_type: type):
    val = untrusted_dict[src_key]
    assert isinstance(val, expected_type), f"assert_type_and_unpack assertion failed on key: {src_key}"
    return val

class EvalWorker:
    def __init__(self, tx_dir: Path, rx_dir: Path, scratch_dir: Path):
        self.tx_dir = tx_dir
        self.rx_dir = rx_dir
        self.scratch_dir = scratch_dir

        self.code_dir = self.scratch_dir / "code"
        os.makedirs(self.code_dir, exist_ok=True)

        self.eval_output_dir = self.scratch_dir / "eval_output"
        os.makedirs(self.eval_output_dir, exist_ok=True)

        self.gpu_locks_dir = self.scratch_dir / "gpu_locks"
        os.makedirs(self.gpu_locks_dir, exist_ok=True)

        self.torch_extensions_dir = self.scratch_dir / "torch_ext"
        os.makedirs(self.torch_extensions_dir, exist_ok=True)

        # remove any existing files in the eval output dir
        for file in self.eval_output_dir.iterdir():
            if file.is_file():
                file.unlink()

        self.disk_channel = DiskChannel(tx_dir, rx_dir)

        self.eval_script_path = curr_dir / "eval.py"

        self.task_count = 0
    
    async def handle_msg(self, msg, task_number):
        level = msg["level"]
        task = msg["task"]
        code_str = msg["code"]
        eval_id = msg["id"]

        # 1. write the LLM-generated code to scratch dir with a unique name
        code_path = self.code_dir / f"task_{level}_{task}_{task_number}.py"
        async with aiofiles.open(code_path, 'w', encoding='utf-8') as f:
            await f.write(code_str)
        
        # print(f"Wrote: {code_path}")
        print(f"Received task: {eval_id}, number: {task_number}")

        eval_output_path = self.eval_output_dir / f"task_{level}_{task}_{task_number}.json"

        # 2. invoke scripts/eval.py with the level, task, and path to the LLM-generated code
        #    (do not wait for this to finish, keep listening for tasks to start)

        eval_torch_ext_dir = self.torch_extensions_dir / str(task_number)
        os.makedirs(eval_torch_ext_dir, exist_ok=True)

        env = os.environ.copy()
        env["TORCH_CUDA_ARCH_LIST"] = "Ampere"
        env["TORCH_EXTENSIONS_DIR"] = str(eval_torch_ext_dir)

        task_start_time = time.time()

        # 10 minutes
        timeout_sec = 10 * 60

        stdout = None
        stderr = None
        timed_out = False

        cmd = [
            "python", str(self.eval_script_path),
            "--level", str(level),
            "--task", str(task),
            "--code_path", str(code_path),
            "--output_path", str(eval_output_path),
            "--gpu_locks_dir", str(self.gpu_locks_dir)]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )

            stdout_raw, stderr_raw = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
            task_end_time = time.time()

            stdout = stdout_raw.decode()

            # print(f"[stdout]\n{stdout}")
            if stderr_raw:
                stderr = stderr_raw.decode()
                # print(f"[stderr]\n{stderr}")

        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            
            timed_out = True

        # 3. read results back

        eval_results = {}

        # TODO: the file may not exist if the eval.py script failed, so maybe we should check if the file exists
        # first to handle things more gracefully
        try:
            async with aiofiles.open(eval_output_path, encoding='utf-8') as f:
                content = await f.read()
            data = json.loads(content)

            model_key = "llm"
            if model_key in data:
                model_res = data[model_key]

                eval_results["loaded"] = assert_type_and_unpack(model_res, "loaded", bool)
                eval_results["correct"] = assert_type_and_unpack(model_res, "correct", bool)
                eval_results["max_diff"] = assert_type_and_unpack(model_res, "max_diff", float)

                if "runtimes" in model_res:
                    runtimes_dict = model_res["runtimes"]
                    eval_results["runtime"] = assert_type_and_unpack(runtimes_dict, "eager", float)
        except Exception as e:
            print(e)

        # 4. send results out to the disk_channel

        output_data = {
            "id": eval_id,
            "results": {
                "stdout": stdout,
                "stderr": stderr,
                "eval_results": eval_results,
                "timed_out": timed_out,
            }
        }

        await self.disk_channel.send(output_data)

        task_time = task_end_time - task_start_time

        print(f"Completed task: {eval_id}, task time: {task_time:.2f}s")

    async def run(self):
        print("Eval worker running...")

        while True:
            msg = await self.disk_channel.recv()
            # print(f"Got message: {msg}")
            asyncio.create_task(self.handle_msg(msg, self.task_count))
            # await self.handle_msg(msg)
            self.task_count += 1
            

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
