import csv
import os
import subprocess
import sys
import time
from datetime import datetime

BUILD_SCRIPT = os.path.join(os.path.dirname(__file__), "build_perf.sh")
BUILD_DIR = "cmake-build-Release"
TEST_DIR = "perf_tests"
RUNS = 500
WARMUPS = 5

TESTS = {
    "perf1": "Large vector ops",
    "perf2": "Parallel sum reduction",
    "perf3": "Memory copy and scaling",
    "perf4": "Realistic high-end compute (typical HPC/ML inner loops)",
    "perf5": "Synthetic memory stress test (~4 GB total global traffic)",
    "perf6": "Synthetic multi-allocation test (1000 buffers, modifies last buffer, ~1 GB traffic)",
}

print("Building all tests...\n")
subprocess.run(["bash", BUILD_SCRIPT], check=True)
print("Build complete.\n")

LD_PRELOAD = os.path.join(BUILD_DIR, "libsafecuda.so")
ENV = os.environ.copy()
ENV["LD_PRELOAD"] = LD_PRELOAD
ENV["LD_LIBRARY_PATH"] = BUILD_DIR

os.makedirs(TEST_DIR + "/results", exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
csv_path = os.path.join(TEST_DIR, "results", f"perf_results_{timestamp}.csv")

print(
    f"Each executable will run {WARMUPS} warm-up runs (ignored) + {RUNS} measured runs.\n"
)

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["test", "description", "compiler", "run", "time_ms"])

    for test, desc in TESTS.items():
        print(
            f"\n------------------------------------ {test}: {desc} ------------------------------------"
        )

        for compiler in ["nvcc", "sfnvcc"]:
            exe = os.path.join(TEST_DIR, f"{test}_{compiler}")
            if not os.path.exists(exe):
                print(f"  [!] Skipping {exe}, not found.")
                continue

            print(f"Running {compiler} ({WARMUPS} warm-ups, {RUNS} measured)...")

            for _ in range(WARMUPS):
                subprocess.run(
                    [exe], env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )

            times = []

            for i in range(RUNS):
                t0 = time.perf_counter()
                subprocess.run(
                    [exe], env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                dt = (time.perf_counter() - t0) * 1000
                times.append(dt)
                avg = sum(times) / len(times)

                sys.stdout.write(
                    f"\r  Last: {dt:7.2f} ms | Avg: {avg:7.2f} ms ({i + 1}/{RUNS})"
                )
                sys.stdout.flush()

                writer.writerow([test, desc, compiler, i, f"{dt:.3f}"])
                f.flush()

            sys.stdout.write("\n")
        print("")

print(f"\nResults saved to {csv_path}")
