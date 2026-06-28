"""Run Module 4: Statistical Testing Pipeline."""
import subprocess, sys, os

BASE = os.path.dirname(__file__)

def run(script):
    result = subprocess.run([sys.executable, os.path.join(BASE, script)], capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"{script} failed")

run("create_master_data.py")
run("module4_stats.py")
print("\n✓ MODULE 4 COMPLETE")
