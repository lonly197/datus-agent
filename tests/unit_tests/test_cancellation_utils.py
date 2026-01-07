import asyncio
import unittest
from datus.utils.async_utils import await_cancellable, ensure_not_cancelled

class TestAsyncUtils(unittest.IsolatedAsyncioTestCase):
    async def test_await_cancellable_success(self):
        async def fast_task():
            return "done"
        result = await await_cancellable(fast_task(), timeout=1.0)
        self.assertEqual(result, "done")

    async def test_await_cancellable_timeout(self):
        async def slow_task():
            await asyncio.sleep(2.0)
            return "done"
        with self.assertRaises(asyncio.TimeoutError):
            await await_cancellable(slow_task(), timeout=0.1)

    async def test_await_cancellable_cancellation(self):
        async def slow_task():
            await asyncio.sleep(2.0)
            return "done"
        
        task = asyncio.create_task(await_cancellable(slow_task(), timeout=5.0))
        await asyncio.sleep(0.1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_ensure_not_cancelled(self):
        async def check_task():
            ensure_not_cancelled()
            await asyncio.sleep(0.1)
            ensure_not_cancelled()
            return "ok"

        t = asyncio.create_task(check_task())
        await asyncio.sleep(0.05)
        t.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await t

if __name__ == '__main__':
    unittest.main()
