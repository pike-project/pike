# worker.py
import asyncio
import os
import shutil
import sys
from pathlib import Path
from filelock import FileLock, Timeout

# --- Setup Paths ---
curr_dir = Path(os.path.realpath(os.path.dirname(__file__)))
locks_dir = curr_dir / "locks"
lock_file = locks_dir / "my_test.lock"
path_to_eval_py = curr_dir / "eval.py"

async def main():
    """
    Runs the full test procedure for the file lock.
    """
    # --- Initial Cleanup ---
    if os.path.exists(locks_dir):
        shutil.rmtree(locks_dir)
    os.makedirs(locks_dir)
    
    print(f"Testing with lock file: {lock_file}")
    
    # ======================================================================
    # Test 1: Confirm lock can be claimed with no contention
    # ======================================================================
    print("\n--- Test 1: Acquiring lock with no contention ---")
    try:
        lock = FileLock(lock_file)
        with lock.acquire(timeout=0.1):
            print("SUCCESS: Lock acquired and released successfully.")
        assert not lock.is_locked
    except Timeout:
        print("FAILURE: Could not acquire an uncontended lock.")
        sys.exit(1)

    # ======================================================================
    # Test 2: Confirm lock CANNOT be claimed if eval.py is running
    # ======================================================================
    print("\n--- Test 2: Testing lock contention ---")
    proc = None
    try:
        # Start eval.py as a subprocess to claim the lock
        cmd = [sys.executable, str(path_to_eval_py), str(lock_file)]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        print(f"Started eval.py with PID: {proc.pid}")

        # IMPORTANT: Wait for eval.py to signal it has the lock
        try:
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=5)
            if b"Lock acquired." not in line:
                raise RuntimeError("eval.py did not report acquiring the lock.")
            print("Confirmed that eval.py has acquired the lock.")
        except asyncio.TimeoutError:
            print("FAILURE: Timed out waiting for eval.py to acquire the lock.")
            proc.kill()
            await proc.wait()
            sys.exit(1)

        # Now, try to claim the lock. This should fail.
        contention_lock = FileLock(lock_file)
        try:
            print("Attempting to acquire the locked file (should time out)...")
            contention_lock.acquire(timeout=1)
            # If we get here, the test failed because we acquired a lock we shouldn't have.
            print("FAILURE: Acquired a lock that should have been held by another process.")
            sys.exit(1)
        except Timeout:
            print("SUCCESS: Timed out as expected. The lock is held by another process.")

        # ======================================================================
        # Test 3: Confirm lock can be claimed after eval.py is killed
        # ======================================================================
        print("\n--- Test 3: Acquiring lock after subprocess is killed ---")
        print(f"Killing eval.py (PID: {proc.pid})...")
        proc.kill()
        await proc.wait()
        print("Subprocess terminated.")
        
        # The lock should now be free.
        try:
            with contention_lock.acquire(timeout=1):
                print("SUCCESS: Acquired the lock after the holder process was killed.")
            assert not contention_lock.is_locked
        except Timeout:
            print("FAILURE: Could not acquire lock even after killing the holder process.")
            sys.exit(1)

    finally:
        # Ensure the subprocess is always killed, even if a test fails
        if proc and proc.returncode is None:
            print("Cleaning up dangling subprocess...")
            proc.kill()
            await proc.wait()

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    # Ensure eval.py exists
    if not os.path.exists(path_to_eval_py):
        print(f"Error: eval.py not found at {path_to_eval_py}")
        sys.exit(1)
        
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\nAn unexpected error occurred during the test: {e}")
        sys.exit(1)
