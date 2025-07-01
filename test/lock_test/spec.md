
I want to write a test for the following (involving 2 Python files):

- `eval.py` - Claims a lock within `locks_dir` indefinitely, a directory shown in the worker outline
- `worker.py` - Does the actual test with the following procedure:
    - starts `eval.py` as a subprocess
    - confirms that the lock can be claimed immediately with no contention
    - confirms that the lock cannot be claimed if `eval.py` is running
    - confirms that the lock can be claimed after `eval.py` times out (as in, the lock is not held indefinitely if `eval.py` is killed)

It is acceptable to use timing to try and confirm these things. `eval.py` can be run many times if needed.

Starting point of `worker.py`:

```python
curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))
locks_dir = curr_dir / "locks"

if os.path.exists(locks_dir):
    shutil.rmtree(locks_dir)

os.makedirs(locks_dir)

timeout_sec = 10

cmd = ["python", path_to_eval_py]

try:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    stdout_raw, stderr_raw = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)

except asyncio.TimeoutError:
    proc.kill()
    await proc.wait()
```
