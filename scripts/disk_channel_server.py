import asyncio
import uuid
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from src.util.disk_channel import DiskChannel
from functools import partial
from urllib.parse import urlparse, parse_qs

class DiskChannelManager:
    def __init__(self, tx_dir: Path, rx_dir: Path):
        self.disk_channel = DiskChannel(tx_dir, rx_dir)

        self.pending_tasks = {}

    async def submit(self, code, problem_id, level):
        eval_id = uuid.uuid4()

        print(eval_id)

        # await self.disk_channel.send({
        #     "id": eval_id,
        #     "type": "eval",
        #     "level": level,
        #     "task": problem_id,
        #     "code": code,
        #     "mode": "eager"
        # })

class CustomHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, manager: DiskChannelManager, **kwargs):
        self.manager = manager

        super().__init__(*args, **kwargs)

    def do_GET(self):
        manager = self.manager

        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query_params = parse_qs(parsed_url.query)

        if path == "/submit":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Welcome to the root!")

            asyncio.run(manager.submit("abc", 1, 1))

        elif path == "/hello":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Hello world!")

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

if __name__ == "__main__":
    p1 = Path("p1")
    p2 = Path("p2")

    manager = DiskChannelManager(tx_dir=p1, rx_dir=p2)

    handler_with_args = partial(CustomHandler, manager=manager)

    server = HTTPServer(("localhost", 8000), handler_with_args)
    print("Serving on http://localhost:8000")
    server.serve_forever()
