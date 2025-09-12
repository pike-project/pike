import asyncio
import uuid
import json
import os
import argparse
import signal
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from src.util.disk_channel import DiskChannel
from functools import partial
from urllib.parse import urlparse, parse_qs, unquote


class DiskChannelManager:
    def __init__(self, tx_dir: Path, rx_dir: Path):
        self.disk_channel = DiskChannel(tx_dir, rx_dir)
        self.completed_tasks = {}
        self._stopping = False

    async def submit(self, code, level, task):
        eval_id = uuid.uuid4()
        await self.disk_channel.send({
            "id": str(eval_id),
            "type": "eval",
            "level": level,
            "task": task,
            "code": code,
            "mode": "eager"
        })
        return eval_id

    async def recv_loop(self):
        while not self._stopping:
            try:
                msg = await self.disk_channel.poll()
                if msg is not None:
                    self.completed_tasks[msg["id"]] = msg
            except asyncio.CancelledError:
                break

            await asyncio.sleep(1.0)

    def stop(self):
        self._stopping = True

    def poll(self, eval_id):
        return self.completed_tasks.get(eval_id)


class CustomHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, manager: DiskChannelManager, loop, **kwargs):
        self.manager = manager
        self.loop = loop
        super().__init__(*args, **kwargs)

    def do_GET(self):
        manager = self.manager
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query_params = parse_qs(parsed_url.query)

        if path == "/submit":
            try:
                code = unquote(query_params.get("code")[0])
                level = str(query_params.get("level")[0])
                task = int(query_params.get("task")[0])
            except Exception:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"Bad input")
                return

            fut = asyncio.run_coroutine_threadsafe(
                manager.submit(code, level, task), self.loop
            )
            eval_id = fut.result()

            self.send_response(200)
            self.end_headers()
            self.wfile.write(f"{eval_id}".encode())

        elif path == "/poll":
            try:
                eval_id = query_params.get("id")[0]
            except Exception:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"Bad input")
                return

            data = manager.poll(eval_id)
            data_str = json.dumps(data)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data_str.encode())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=False, default=8000)
    parser.add_argument("--worker_io_dir", type=str, required=False, default="worker_io")
    args = parser.parse_args()

    tx_dir = Path(args.worker_io_dir) / "input"
    rx_dir = Path(args.worker_io_dir) / "output"
    os.makedirs(tx_dir, exist_ok=True)
    os.makedirs(rx_dir, exist_ok=True)

    manager = DiskChannelManager(tx_dir=tx_dir, rx_dir=rx_dir)

    recv_task = asyncio.create_task(manager.recv_loop())

    loop = asyncio.get_running_loop()
    handler_with_args = partial(CustomHandler, manager=manager, loop=loop)

    server = ThreadingHTTPServer(("localhost", args.port), handler_with_args)

    stop_event = asyncio.Event()

    def handle_stop(*_):
        stop_event.set()

    loop.add_signal_handler(signal.SIGINT, handle_stop)
    loop.add_signal_handler(signal.SIGTERM, handle_stop)

    # Run HTTP server in a background thread until stop_event is set
    server_task = asyncio.create_task(asyncio.to_thread(server.serve_forever))

    await stop_event.wait()

    # Shutdown everything
    print("Shutting down...")
    server.shutdown()
    manager.stop()
    recv_task.cancel()

    try:
        await recv_task
    except asyncio.CancelledError:
        pass

    # Wait for HTTP server thread to exit
    await server_task

    print("Exited cleanly")


if __name__ == "__main__":
    asyncio.run(main())
