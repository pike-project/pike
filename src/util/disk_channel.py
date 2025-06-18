import asyncio
from pathlib import Path

# The DiskChannel uses async file I/O methods to do the following
# (without watchdog or other deps since this is used on an NFS system)
# - recv() waits until there is a new "done" flag file in rx_dir, indicating the data is ready
#   - the "done" flag only indicates the json data file is ready, but the actual data will be in the corresponding json file
#   - if many new "done" files appear between subsequent checks of the directory, only return one of the corresponding
#     data files on this iteration, but be sure to return the other data files on future iterations
# - send() first writes the actual json data to a file, then writes the "done" flag file only when writing of the data is complete
#   - each data/done pair should have a unique ID, so that outgoing/incoming data files do not clash

class DiskChannel:
    def __init__(self, tx_dir: Path, rx_dir: Path):
        self.tx_dir = tx_dir
        self.rx_dir = rx_dir

    # polling async recv method that waits until there is a new message
    # TODO: could add timeout parameter, and other utilities, but this is
    # the barebones that we need
    async def recv(self):
        while True:
            # this should eventually return when a new "done" file appears in rx_dir
            await asyncio.sleep(1)
        # return {}

    # TODO: send out data via tx_dir
    def send(self, data: dict):
        pass
