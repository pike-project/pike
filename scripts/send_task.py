import sys
import os
import asyncio
import uuid
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.util.disk_channel import DiskChannel

async def main():
    tx_dir = Path("worker_io/input")
    rx_dir = Path("worker_io/output")

    task_id = str(uuid.uuid4())
    print(f"Sent task: {task_id}")

    with open("results/o3-test1/generated_kernel_level_1_problem_1.py") as f:
        code = f.read()

    disk_channel = DiskChannel(tx_dir, rx_dir)
    await disk_channel.send({
        "id": task_id,
        "level": 1,
        "task": 1,
        "code": code
    })

    print(await disk_channel.recv())

if __name__ == "__main__":
    asyncio.run(main())
