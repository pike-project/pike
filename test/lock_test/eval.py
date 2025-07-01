# eval.py
import sys
import time
from pathlib import Path
from filelock import FileLock

def main():
    """
    Acquires a lock and holds it indefinitely until the process is killed.
    """
    if len(sys.argv) < 2:
        print("Error: Please provide the path to the lock file.", file=sys.stderr)
        sys.exit(1)

    lock_path = Path(sys.argv[1])
    lock = FileLock(lock_path)

    try:
        # Acquire the lock. This will block until the lock is available.
        lock.acquire()
        
        # Signal to the parent process that the lock has been acquired.
        # Flushing is essential so the parent receives the message immediately.
        print("Lock acquired.", flush=True)

        # Hold the lock "indefinitely"
        while True:
            time.sleep(1)

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
    finally:
        # This part will only be reached if the loop is broken,
        # but the OS will clean up the lock when the process is killed.
        if lock.is_locked:
            lock.release()
            print("Lock released.", flush=True)

if __name__ == "__main__":
    main()
