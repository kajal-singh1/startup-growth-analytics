"""Run Module 5: ML Pipeline."""
import subprocess, sys, os

BASE = os.path.dirname(__file__)

def run(script):
    result = subprocess.run([sys.executable, os.path.join(BASE, script)])
    if result.returncode != 0:
        raise RuntimeError(f"{script} failed")

run("create_master_data.py")
run("module5_ml.py")
print("\n✓ MODULE 5 COMPLETE")
