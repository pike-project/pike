import asyncio
import uuid
import json
import logging
import os
import argparse
import signal
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from src.util.disk_channel import DiskChannel
from functools import partial
from urllib.parse import urlparse, parse_qs, unquote

logger = logging.getLogger("disk_channel_server")


class DiskChannelManager:
    def __init__(self, tx_dir: Path, rx_dir: Path, verbose: bool = False):
        self.disk_channel = DiskChannel(tx_dir, rx_dir, verbose=verbose)
        self.completed_tasks = {}
        self.handshake_complete = False
        self._stopping = False

    async def start_handshake(self):
        await self.disk_channel.send({
            "type": "handshake",
        })

    async def submit(self, code, level, task, mode="eager"):
        eval_id = uuid.uuid4()
        logger.debug("submit(): sending eval level=%s task=%s mode=%s eval_id=%s code_len=%d",
                      level, task, mode, eval_id, len(code))
        await self.disk_channel.send({
            "id": str(eval_id),
            "type": "eval",
            "level": level,
            "task": task,
            "code": code,
            "mode": mode,
        })
        logger.debug("submit(): disk_channel.send() completed for eval_id=%s", eval_id)
        return eval_id

    async def recv_loop(self):
        await self.start_handshake()

        while not self._stopping:
            try:
                msg = await self.disk_channel.poll()
                if msg is not None:
                    msg_type = msg["type"]
                    if msg_type == "handshake":
                        self.handshake_complete = True
                        logger.debug("recv_loop(): handshake complete")
                    elif msg_type == "result":
                        self.completed_tasks[msg["id"]] = msg
                        logger.debug("recv_loop(): received result for eval_id=%s (total stored: %d)",
                                     msg["id"], len(self.completed_tasks))
            except asyncio.CancelledError:
                break

            await asyncio.sleep(1.0)

    def stop(self):
        self._stopping = True

    def poll(self, eval_id):
        result = self.completed_tasks.get(eval_id)
        if result is not None:
            logger.debug("poll(): found result for eval_id=%s", eval_id)
        return result

    async def close(self):
        print("Sending disk_channel close message")
        await self.disk_channel.send({
            "type": "close"
        })

class CustomHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, manager: DiskChannelManager, loop, verbose: bool = False, **kwargs):
        self.manager = manager
        self.loop = loop
        self.verbose = verbose
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        pass
        # if self.verbose:
        #     super().log_message(format, *args)

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
                mode = str(query_params.get("mode", ["eager"])[0])
            except Exception:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"Bad input")
                return

            logger.debug("/submit: level=%s task=%d mode=%s code_len=%d", level, task, mode, len(code))

            fut = asyncio.run_coroutine_threadsafe(
                manager.submit(code, level, task, mode), self.loop
            )
            eval_id = fut.result()

            logger.debug("/submit: returned eval_id=%s for task=%d", eval_id, task)

            self.send_response(200)
            self.end_headers()
            self.wfile.write(f"{eval_id}".encode())
        elif path == "/ready":
            data_str = json.dumps(manager.handshake_complete)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data_str.encode())
        elif path == "/poll":
            try:
                eval_id = query_params.get("id")[0]
            except Exception:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"Bad input")
                return

            data = manager.poll(eval_id)
            logger.debug("/poll: eval_id=%s found=%s", eval_id, data is not None)
            data_str = json.dumps(data)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data_str.encode())
        elif path == "/close":
            fut = asyncio.run_coroutine_threadsafe(
                manager.close(), self.loop
            )
            _ = fut.result()

            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=False, default=8000, help="Port to serve this HTTP server on")
    parser.add_argument("--worker-io-dir", type=str, required=False, default="worker_io", help="Scratch directory for communicating with the worker via filesystem")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(name)s] %(message)s",
        )

    tx_dir = Path(args.worker_io_dir) / "input"
    rx_dir = Path(args.worker_io_dir) / "output"
    os.makedirs(tx_dir, exist_ok=True)
    os.makedirs(rx_dir, exist_ok=True)

    manager = DiskChannelManager(tx_dir=tx_dir, rx_dir=rx_dir, verbose=args.verbose)

    recv_task = asyncio.create_task(manager.recv_loop())

    loop = asyncio.get_running_loop()
    handler_with_args = partial(CustomHandler, manager=manager, loop=loop, verbose=args.verbose)

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
