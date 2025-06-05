import asyncio
from pathlib import Path

class DiskChannel:
    def __init__(self, tx_dir: Path, rx_dir: Path):
        self.tx_dir = tx_dir
        self.rx_dir = rx_dir

    # polling async recv method that waits until there is a new message
    # TODO: could add timeout parameter, and other utilities, but this is
    # the barebones that we need
    async def recv(self):
        await asyncio.sleep(10)
        return {}

    def send(self):
        pass
