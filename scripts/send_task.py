import sys
import os
import asyncio
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.util.disk_channel import DiskChannel

async def main():
    tx_dir = Path("worker_io/input")
    rx_dir = Path("worker_io/output")

    disk_channel = DiskChannel(tx_dir, rx_dir)

    await disk_channel.send({
        "level": 1,
        "task": 1,
        "code": None
    })

if __name__ == "__main__":
    asyncio.run(main())
