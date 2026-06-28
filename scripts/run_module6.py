import subprocess
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run(script):
    print(f"Running {script}...")
    result = subprocess.run([sys.executable, os.path.join(BASE_DIR, script)])
    if result.returncode != 0:
        raise RuntimeError(f"{script} failed")


if __name__ == "__main__":
    run("module6_forecast.py")
