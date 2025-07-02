import asyncio

class AtomicIntAsync:
    def __init__(self, value=0):
        self._value = value
        self._lock = asyncio.Lock()

    async def add(self, delta):
        async with self._lock:
            self._value += delta
            return self._value

    async def inc(self):
        return await self.add(1)

    async def dec(self):
        return await self.add(-1)

    async def get(self):
        async with self._lock:
            return self._value

    def peek(self):
        # UNSAFE read — no lock!
        return self._value
