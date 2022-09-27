import time
import sys
import signal


def ignore_handler(signum, frame):
    print("Ignoring SIGTERM")


def main(ignore_term=False, wait_time=-1):
    print(f"Call with {ignore_term}, {wait_time}")
    if ignore_term:
        signal.signal(signal.SIGTERM, ignore_handler)
    if wait_time > 0:
        time.sleep(wait_time)
    else:
        while True:
            pass


if __name__ == "__main__":
    main(*[float(x) for x in sys.argv[1:]])
