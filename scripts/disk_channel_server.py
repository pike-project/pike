import asyncio
import uuid
import json
import argparse
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from src.util.disk_channel import DiskChannel
from functools import partial
from urllib.parse import urlparse, parse_qs, quote, unquote

class DiskChannelManager:
    def __init__(self, tx_dir: Path, rx_dir: Path):
        self.disk_channel = DiskChannel(tx_dir, rx_dir)

        self.completed_tasks = {}

    async def submit(self, code, level, task):
        eval_id = uuid.uuid4()

        print(eval_id)

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
        while True:
            msg = await self.disk_channel.recv()
            self.completed_tasks[msg["id"]] = msg

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

        print(query_params)

        if path == "/submit":
            try:
                code = unquote(query_params.get("code")[0])
                level = int(query_params.get("level")[0])
                task = int(query_params.get("task")[0])
            except Exception as e:
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
            except Exception as e:
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
    parser.add_argument("--input_dir", type=str, default="worker_io/input")
    parser.add_argument("--output_dir", type=str, default="worker_io/output")
    args = parser.parse_args()

    tx_dir = Path(args.input_dir)
    rx_dir = Path(args.output_dir)

    manager = DiskChannelManager(tx_dir=tx_dir, rx_dir=rx_dir)

    asyncio.create_task(manager.recv_loop())

    loop = asyncio.get_running_loop()

    handler_with_args = partial(CustomHandler, manager=manager, loop=loop)

    server = ThreadingHTTPServer(("localhost", 8000), handler_with_args)
    print("Serving on http://localhost:8000")
    await asyncio.to_thread(server.serve_forever)

if __name__ == "__main__":
    asyncio.run(main())
