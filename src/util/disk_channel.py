import asyncio
import json
import uuid
from pathlib import Path
from collections import deque

# aiofiles is a library for asynchronous file operations.
# It's a good choice for this problem as it integrates well with asyncio.
# To install: pip install aiofiles
import aiofiles

# The DiskChannel uses async file I/O methods to do the following
# (without watchdog or other deps since this is used on an NFS system)
# - recv() waits until there is a new "done" flag file in rx_dir, indicating the data is ready
#   - the "done" flag only indicates the json data file is ready, but the actual data will be in the corresponding json file
#   - if many new "done" files appear between subsequent checks of the directory, only return one of the corresponding
#     data files on this iteration, but be sure to return the other data files on future iterations
# - send() first writes the actual json data to a file, then writes the "done" flag file only when writing of the data is complete
#   - each data/done pair should have a unique ID, so that outgoing/incoming data files do not clash

class DiskChannel:
    """
    A communication channel that uses the filesystem for message passing.

    This is useful for inter-process communication on systems where standard
    IPC mechanisms are difficult to use, such as across different containers
    or on a Network File System (NFS).

    Messages are sent by writing a JSON data file and then an empty ".done"
    file to signal that the data is ready. Messages are received by polling
    for ".done" files, reading the corresponding JSON file, and then deleting
    both files.
    """
    def __init__(self, tx_dir: Path, rx_dir: Path, poll_interval: float = 1.0):
        """
        Initializes the DiskChannel.

        Args:
            tx_dir: The directory to write outgoing messages to.
            rx_dir: The directory to read incoming messages from.
            poll_interval: The time in seconds to wait between polling for
                           new messages when the queue is empty.
        """
        self.tx_dir = tx_dir
        self.rx_dir = rx_dir
        self.poll_interval = poll_interval

        # Create directories if they don't exist
        self.tx_dir.mkdir(parents=True, exist_ok=True)
        self.rx_dir.mkdir(parents=True, exist_ok=True)

        # remove any existing files in the sending channel
        for file in self.tx_dir.iterdir():
            if file.is_file():
                file.unlink()

        # Internal queue for discovered messages to ensure they are processed one by one.
        # We store the path to the .done file.
        self._pending_files = deque()
        
        # A set to keep track of .done files that have been discovered but not yet
        # processed. This prevents adding the same file to the queue multiple times
        # and its size is proportional to the backlog, preventing memory leaks.
        self._seen_files = set()

    async def poll(self):
        # If the internal queue is empty, scan the directory for new files.
        if not self._pending_files:
            self._scan_for_new_files()

        # If the queue has files, process the first one.
        if self._pending_files:
            done_path = self._pending_files.popleft()
            
            try:
                data = await self._process_file(done_path)
                # On successful processing, remove from seen set and return
                self._seen_files.remove(done_path)
                return data
            except Exception as e:
                print(f"Error processing {done_path.name}: {e}. Discarding.")
                # Ensure the file is removed from tracking even on error
                if done_path in self._seen_files:
                    self._seen_files.remove(done_path)
                # Continue to the next file in the queue or re-poll
                return None

        return None

    # polling async recv method that waits until there is a new message
    async def recv(self) -> dict:
        """
        Waits for and receives a message from the channel.

        This method polls the `rx_dir` for new ".done" files. If multiple
        files are found, they are queued internally and processed one by one
        on subsequent calls to `recv`.

        Returns:
            A dictionary containing the message data.
        """
        while True:
            res = await self.poll()

            if res is not None:
                return res
            
            # If queue is still empty after polling, wait before polling again.
            await asyncio.sleep(self.poll_interval)
            
    def _scan_for_new_files(self):
        """Synchronously scans the rx_dir for new .done files and adds them to the queue."""
        try:
            # Sort by modification time to process in a predictable, chronological order.
            # Note: list(glob(...)) is synchronous and can block on very large directories
            # or slow filesystems. For most cases, this is acceptable.
            all_done_files = sorted(
                self.rx_dir.glob('*.done'), 
                key=lambda p: p.stat().st_mtime
            )
            for done_file in all_done_files:
                if done_file not in self._seen_files:
                    self._seen_files.add(done_file)
                    self._pending_files.append(done_file)
        except FileNotFoundError:
            # The rx_dir might have been deleted. It will be recreated on next send.
            print(f"Warning: rx_dir {self.rx_dir} not found during scan.")
        except Exception as e:
            print(f"Error scanning directory {self.rx_dir}: {e}")

    async def _process_file(self, done_path: Path) -> dict:
        """Reads, decodes, and cleans up a single message file pair."""
        data_path = done_path.with_suffix('.json')

        # Run synchronous file check in a thread to avoid blocking.
        if not await asyncio.to_thread(data_path.is_file):
            # The data file is missing, an error state. Clean up the .done file.
            await asyncio.to_thread(done_path.unlink, missing_ok=True)
            raise FileNotFoundError(f"Data file {data_path} not found for done file {done_path}")

        try:
            async with aiofiles.open(data_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            data = json.loads(content)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            # If file is unreadable or disappears, clean up both and re-raise.
            await asyncio.to_thread(data_path.unlink, missing_ok=True)
            await asyncio.to_thread(done_path.unlink, missing_ok=True)
            raise e

        # Clean up both files after successful read.
        await asyncio.to_thread(data_path.unlink, missing_ok=True)
        await asyncio.to_thread(done_path.unlink, missing_ok=True)
        
        return data

    # Note: The original stub had a synchronous `send` method.
    # It has been made asynchronous (`async def`) to use non-blocking file I/O,
    # which is crucial for performance in an asyncio application.
    async def send(self, data: dict):
        """
        Sends a message through the channel.

        This operation is performed in two steps to ensure atomicity for the
        receiver:
        1. The data is written to a unique ".json" file.
        2. An empty ".done" file with the same unique name is created to signal
           that the data file is complete and ready for processing.

        Args:
            data: The dictionary data to send.
        """
        message_id = str(uuid.uuid4())
        data_path = self.tx_dir / f"{message_id}.json"
        done_path = self.tx_dir / f"{message_id}.done"

        # 1. Write the data to the JSON file.
        async with aiofiles.open(data_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(data))

        # 2. Create the "done" flag file to signal completion.
        # This is the atomic signal that the message is ready.
        async with aiofiles.open(done_path, 'w'):
            pass


# Example usage to demonstrate and test the DiskChannel
async def main():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        base_dir = Path(d)
        comm_dir = base_dir / "comm_channel"
        
        print(f"Using temporary directory: {comm_dir}")

        # In this example, one process sends and receives from the same directory.
        channel = DiskChannel(tx_dir=comm_dir, rx_dir=comm_dir)

        async def test_runner():
            print("\n--- Test 1: Simple send and receive ---")
            msg_to_send = {"id": 1, "text": "Testing 1-2-3"}
            print(f"Sending: {msg_to_send}")
            await channel.send(msg_to_send)
            
            print("Receiving...")
            received_msg = await channel.recv()
            print(f"Received: {received_msg}")
            assert received_msg == msg_to_send
            print("Test 1 PASSED")

            print("\n--- Test 2: Queuing behavior ---")
            print("Sending 3 messages in quick succession...")
            for i in range(3):
                await channel.send({"message_index": i})
            
            print("Receiving 3 messages one by one...")
            received_indices = []
            for i in range(3):
                data = await channel.recv()
                print(f"Received message {i+1}: {data}")
                received_indices.append(data["message_index"])
            
            assert sorted(received_indices) == [0, 1, 2]
            print("Test 2 PASSED")

            print("\n--- Test 3: Concurrent send/receive ---")
            async def concurrent_sender():
                for i in range(5):
                    await channel.send({"concurrent_msg": i})
                    await asyncio.sleep(0.1)

            async def concurrent_receiver():
                received_count = 0
                while received_count < 5:
                    data = await channel.recv()
                    print(f"Concurrent recv: {data}")
                    received_count += 1
                return received_count

            sender_task = asyncio.create_task(concurrent_sender())
            receiver_task = asyncio.create_task(concurrent_receiver())
            
            await sender_task
            count = await receiver_task
            assert count == 5
            print("Test 3 PASSED")


        await test_runner()
        print("\nAll tests completed successfully.")


if __name__ == "__main__":
    # To run this example, make sure you have aiofiles installed:
    # pip install aiofiles
    try:
        asyncio.run(main())
    except ImportError:
        print("\nPlease install 'aiofiles' to run this example: pip install aiofiles")
