import subprocess
import sys
from pathlib import Path
from datetime import datetime
import time


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

STEPS = [
    {
        "name": "UPDATE DAILY DATA",
        "script": "update_daily_data.py",
        "args": [],
    },
    {
        "name": "RUN V2 SCANNER",
        "script": "scanner_v2.py",
        "args": [],
    },
    {
        "name": "TRACK SETUP HISTORY",
        "script": "track_setup_history_v2.py",
        "args": [],
    },
]


# ============================================================
# RUN ONE SCRIPT
# ============================================================

def run_step(step_number, total_steps, step):
    name = step["name"]
    script_name = step["script"]
    args = step.get("args", [])

    script_path = PROJECT_DIR / script_name

    print("\n")
    print("=" * 75)
    print(f"STEP {step_number}/{total_steps}: {name}")
    print("=" * 75)

    if not script_path.exists():
        print(f"ERROR: Script not found:")
        print(script_path)
        return False

    command = [
        sys.executable,
        str(script_path),
        *args
    ]

    print(f"Running: {' '.join(command)}")
    print("-" * 75)

    start_time = time.time()

    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_DIR,
            check=False
        )

        elapsed = time.time() - start_time

        print("-" * 75)

        if result.returncode == 0:
            print(
                f"SUCCESS: {name} completed "
                f"in {elapsed / 60:.1f} minutes."
            )
            return True

        print(
            f"FAILED: {name} returned "
            f"exit code {result.returncode}."
        )
        return False

    except KeyboardInterrupt:
        print("\nProcess stopped by user.")
        raise

    except Exception as exc:
        print(f"\nERROR running {name}:")
        print(exc)
        return False


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print("\n")
    print("=" * 75)
    print("EMA REVERSAL SCANNER - DAILY RUN")
    print("=" * 75)
    print(
        "Started:",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    print("\nSequence:")
    for i, step in enumerate(STEPS, start=1):
        print(f"  {i}. {step['name']}")

    print("=" * 75)

    total_steps = len(STEPS)

    for i, step in enumerate(STEPS, start=1):

        success = run_step(
            step_number=i,
            total_steps=total_steps,
            step=step
        )

        if not success:
            print("\n")
            print("=" * 75)
            print("DAILY RUN STOPPED")
            print("=" * 75)
            print(
                f"Failure occurred during: {step['name']}"
            )
            print(
                "Later steps were NOT executed."
            )
            print("=" * 75)

            sys.exit(1)

    total_elapsed = time.time() - start_time

    print("\n")
    print("=" * 75)
    print("DAILY SCANNER RUN COMPLETE")
    print("=" * 75)
    print(
        "Finished:",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    print(
        f"Total runtime: {total_elapsed / 60:.1f} minutes"
    )
    print("=" * 75)

    print("\nCompleted successfully:")
    for step in STEPS:
        print(f"  ✓ {step['name']}")

    print("\nYour scanner output and tracking files are ready.")


if __name__ == "__main__":
    main()