import time
import logging
import requests

logger = logging.getLogger("disk_channel_client")


class DiskChannelClient:
    def __init__(self, port: int, poll_interval: float = 1.0, poll_timeout: float = 3600.0):
        self.base_url = f"http://localhost:{port}"
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout

    def wait_for_ready(self) -> None:
        """Poll GET /ready until server returns true. No timeout (blocks indefinitely)."""
        while True:
            try:
                res = requests.get(f"{self.base_url}/ready")
                if res.text == "true":
                    return
            except Exception:
                pass
            time.sleep(self.poll_interval)

    def submit(self, code: str, level: str, task: int, mode: str | None = None) -> str:
        """POST-like submit via GET /submit. Returns eval_id string. Raises on failure."""
        params = {"code": code, "level": level, "task": task}
        if mode is not None:
            params["mode"] = mode
        res = requests.get(f"{self.base_url}/submit", params=params)
        res.raise_for_status()
        return res.text

    def poll_for_result(self, eval_id: str) -> dict | None:
        """Poll GET /poll until non-null result or timeout. Returns result dict or None."""
        start_time = time.time()
        poll_count = 0
        while True:
            try:
                res = requests.get(f"{self.base_url}/poll", params={"id": eval_id})
                data = res.json()
                poll_count += 1
                if data is not None:
                    elapsed = time.time() - start_time
                    logger.debug("poll result arrived for eval_id=%s after %d polls (%.1fs)", eval_id, poll_count, elapsed)
                    return data
            except Exception as e:
                print(f"Poll error for {eval_id}: {e}")

            if poll_count % 30 == 0 and poll_count > 0:
                elapsed = time.time() - start_time
                logger.debug("still polling eval_id=%s (%d polls, %.1fs elapsed)", eval_id, poll_count, elapsed)

            if time.time() - start_time > self.poll_timeout:
                print(f"Timeout waiting for eval_id {eval_id}")
                logger.debug("TIMEOUT for eval_id=%s after %d polls", eval_id, poll_count)
                return None

            time.sleep(self.poll_interval)

    def close(self) -> None:
        """Send GET /close to shut down the worker."""
        requests.get(f"{self.base_url}/close")
