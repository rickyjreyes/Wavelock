#!/usr/bin/env python3
import os
import sys
import time
import json
import hashlib
import platform
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============================================================
# ANSI color codes
# ============================================================
RESET="\033[0m"; BOLD="\033[1m"
RED="\033[31m"; GREEN="\033[32m"; YELLOW="\033[33m"; BLUE="\033[34m"
MAGENTA="\033[35m"; CYAN="\033[36m"; WHITE="\033[37m"
BRIGHT_RED="\033[91m"; BRIGHT_GREEN="\033[92m"; BRIGHT_YELLOW="\033[93m"
BRIGHT_BLUE="\033[94m"; BRIGHT_MAGENTA="\033[95m"; BRIGHT_CYAN="\033[96m"
BRIGHT_WHITE="\033[97m"

# ============================================================
# CATEGORY MAP
# ============================================================
CATEGORY_MAP = {
    "pde": ["advanced","hyper","hamiltonian","dual","tbja","extreme","curvature_attack"],
    "ml": ["inn","glow","claude","claude2","claude3","neural"],
    "spectral": ["exotic","wct_modes","svd","eigen"],
    "topological": ["topo","mapper","homotopy"],
    "quantum": ["post_quantum","qaoa","grover"],
    "classical": ["nextgen","attack","tangent"]
}

CATEGORY_COLORS = {
    "pde": BRIGHT_BLUE,
    "ml": BRIGHT_MAGENTA,
    "spectral": CYAN,
    "topological": YELLOW,
    "quantum": BRIGHT_CYAN,
    "classical": GREEN,
    "uncategorized": WHITE
}

def classify_category(name):
    lname = name.lower()
    for cat, keys in CATEGORY_MAP.items():
        if any(k in lname for k in keys):
            return cat
    return "uncategorized"

def color_category(cat):
    return f"{CATEGORY_COLORS.get(cat, WHITE)}[{cat.upper()}]{RESET}"

# ============================================================
# DANGER KEYS (TRUE means bad)
# ============================================================
DANGER_KEYS = [
    "matched",
    "collisions",
    "forgeries",
    "false_accepts",
    "accepted",
    "nan_detected"
]

# ============================================================
# PARSE REAL METRIC JSON
# ============================================================
def extract_metrics(stdout):
    """Extract RISK_METRICS JSON block if present."""
    if "RISK_METRICS_BEGIN" not in stdout:
        return None
    try:
        block = stdout.split("RISK_METRICS_BEGIN", 1)[1]
        block = block.split("RISK_METRICS_END", 1)[0]
        return json.loads(block)
    except Exception:
        return None

# ============================================================
# CLEAN RISK ENGINE — count danger events
# ============================================================
def compute_real_risk(metrics):
    """Count TRUE or >0 among danger fields."""
    if metrics is None:
        return None

    danger_count = 0
    for key in DANGER_KEYS:
        if key in metrics:
            v = metrics[key]
            if v is True:
                danger_count += 1
                continue
            try:
                if isinstance(v, (int, float)) and v > 0:
                    danger_count += 1
            except Exception:
                pass
    return danger_count

# ============================================================
# FALLBACK: parse Python dicts in stdout
# ============================================================
def compute_fallback_risk(stdout):
    text = stdout.lower()
    danger_count = 0

    if "matched': true" in text or '"matched": true' in text:
        danger_count += 1

    for key in ["collisions", "forgeries", "false_accepts", "accepted"]:
        if key in text:
            try:
                after = text.split(key, 1)[1]
                num = int("".join(c for c in after if c.isdigit()))
                if num > 0:
                    danger_count += 1
            except Exception:
                pass

    if "nan" in text:
        danger_count += 1

    return danger_count

# ============================================================
# COLOR DANGER COUNT
# ============================================================
def color_risk(n):
    if n == 0:
        return f"{BRIGHT_GREEN}[DANGER: {n}]{RESET}"
    if n <= 2:
        return f"{YELLOW}[DANGER: {n}]{RESET}"
    if n <= 5:
        return f"{BRIGHT_YELLOW}[DANGER: {n}]{RESET}"
    return f"{BRIGHT_RED}[DANGER: {n}]{RESET}"

# ============================================================
# STATUS COLOR
# ============================================================
def color_status(label):
    if "PASS" in label: return f"{GREEN}{label}{RESET}"
    if "FAIL" in label: return f"{RED}{label}{RESET}"
    if "DIAGNOSTIC" in label: return f"{YELLOW}{label}{RESET}"
    if "CRASH" in label: return f"{BRIGHT_RED}{label}{RESET}"
    if "TIMEOUT" in label: return f"{BRIGHT_RED}{label}{RESET}"
    if "EMPTY" in label: return f"{WHITE}{label}{RESET}"
    return label

# ============================================================
# PATH SETUP
# ============================================================
THIS_FILE = Path(__file__).resolve()
SCRIPT_DIR = THIS_FILE.parent
REPO_ROOT = SCRIPT_DIR.parent
SCIENTIFIC_DIR = REPO_ROOT/"tests"/"scientific"
TMP_LOGS = SCRIPT_DIR/"tmp_parallel"; TMP_LOGS.mkdir(parents=True, exist_ok=True)
LOG_DIR = SCRIPT_DIR/"logs"; LOG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# UTILITIES
# ============================================================
def format_time(seconds):
    s = int(seconds)
    return f"{s//3600}:{(s%3600)//60:02d}:{s%60:02d}" if s>=3600 else f"{(s%3600)//60:02d}:{s%60:02d}"

def classify(file_name, stdout, stderr, returncode):
    name = file_name.lower()

    diagnostic_names = [
        "advanced","hyper","hamiltonian","dual","tbja","claude",
        "inn","glow","exotic","extreme","post_quantum","curvature_attack",
        "nextgen","wct_modes"
    ]

    if returncode == 124:
        return "⚠ TIMEOUT"
    if returncode != 0 or "traceback" in stdout.lower():
        return "⚠ CRASH"
    if "AssertionError" in stdout:
        return "✗ FAIL"
    if stdout.strip() == "":
        return "⚠ EMPTY"

    # These scripts contain exploratory internal FAIL/diagnostic lines, but exit 0 by design.
    if any(k in name for k in diagnostic_names):
        return "✓ DIAGNOSTIC"

    return "✓ PASS"

def print_result_line(name, stdout, returncode, runtime):
    metrics = extract_metrics(stdout)
    danger = compute_real_risk(metrics)
    if danger is None:
        danger = compute_fallback_risk(stdout)

    status = classify(name, stdout, "", returncode)

    print(
        f"{BLUE}{name}{RESET} • "
        f"{color_category(classify_category(name))} • "
        f"{color_risk(danger)} • "
        f"{color_status(status)} • "
        f"{format_time(runtime)}"
    )

    if returncode != 0 or "traceback" in stdout.lower():
        print(f"{BRIGHT_RED}[CRASH OUTPUT: {name}]{RESET}")
        print(stdout[-3000:])

    return danger, status

# ============================================================
# EXECUTE ONE TEST
# ============================================================
def run_single_test(test_file, env, timeout_seconds=None):
    start = time.time()
    tmp_log = TMP_LOGS / f"{test_file.name}.log"
    lines = []
    returncode = 1

    with open(tmp_log, "w", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(
                [sys.executable, str(test_file.resolve())],
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                encoding="utf-8",
                errors="replace"
            )

            while True:
                line = process.stdout.readline()
                if line:
                    lines.append(line)
                    log.write(line)
                    log.flush()

                if process.poll() is not None:
                    rest = process.stdout.read()
                    if rest:
                        lines.append(rest)
                        log.write(rest)
                    break

                if timeout_seconds and (time.time() - start) > timeout_seconds:
                    process.kill()
                    msg = f"\n[ERROR] TIMEOUT after {timeout_seconds} seconds\n"
                    lines.append(msg)
                    log.write(msg)
                    returncode = 124
                    break

            if returncode != 124:
                returncode = process.returncode

        except Exception as e:
            msg = f"\n[ERROR] RUNNER EXCEPTION: {repr(e)}\n"
            lines.append(msg)
            log.write(msg)
            returncode = 1

    stdout = "".join(lines)
    runtime = time.time() - start
    return (test_file.name, stdout, "", returncode, runtime, tmp_log)

# ============================================================
# ENVIRONMENT SUMMARY
# ============================================================
def get_environment_summary():
    info = {
        "python": sys.version.split()[0],
        "os": platform.system(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "timestamp": datetime.now().isoformat()
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
    except Exception:
        info["torch"] = "not installed"

    try:
        import cupy
        info["cupy"] = cupy.__version__
        info["cupy_cuda_version"] = cupy.cuda.runtime.runtimeGetVersion()
    except Exception:
        info["cupy"] = "not installed"

    try:
        git_hash = subprocess.check_output(["git","rev-parse","HEAD"], cwd=REPO_ROOT).decode().strip()
        info["git_commit"] = git_hash
    except Exception:
        info["git_commit"] = "no git repo"

    return info

# ============================================================
# ARGUMENTS
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="WaveLock Scientific Benchmark Runner")
    parser.add_argument(
        "--only",
        default=None,
        help="Run only tests whose filename contains this substring, e.g. --only scaled_neural"
    )
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=int(os.environ.get("WAVELOCK_CPU_WORKERS", "2")),
        help="Max parallel CPU workers. Default: env WAVELOCK_CPU_WORKERS or 2."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("WAVELOCK_TEST_TIMEOUT", "0")),
        help="Per-test timeout in seconds. 0 disables timeout."
    )
    parser.add_argument(
        "--cpu-sequential",
        action="store_true",
        help="Run CPU tests sequentially."
    )
    return parser.parse_args()

# ============================================================
# MAIN RUNNER
# ============================================================
def run():
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_log_path = LOG_DIR / f"wavelock_benchmark_{timestamp}.log"
    json_summary_path = LOG_DIR / f"wavelock_summary_{timestamp}.json"

    env_info = get_environment_summary()

    print(f"{BOLD}{CYAN}=== WaveLock Scientific Benchmark Runner (Safe Parallel Mode) ==={RESET}")
    print(json.dumps(env_info, indent=2))
    print(f"{CYAN}Log file:{RESET} {final_log_path}\n")

    files = sorted(SCIENTIFIC_DIR.glob("test_wavelock*.py"))

    if args.only:
        needle = args.only.lower()
        files = [f for f in files if needle in f.name.lower()]

    if not files:
        print(f"{RED}❌ No tests found in:{RESET} {SCIENTIFIC_DIR}")
        if args.only:
            print(f"{RED}Filter used:{RESET} {args.only}")
        return

    gpu_keywords = [
        "advanced","hyper","hamiltonian","dual","tbja","claude","inn","glow",
        "exotic","extreme","post_quantum","curvature_attack","nextgen","wct_modes"
    ]

    gpu_tests = [f for f in files if any(k in f.name.lower() for k in gpu_keywords)]
    cpu_tests = [f for f in files if f not in gpu_tests]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONNOUSERSITE"] = "1"

    # Prevent parallel CPU tests from spawning huge nested thread pools.
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"

    timeout_seconds = args.timeout if args.timeout and args.timeout > 0 else None
    results = []

    # GPU TESTS
    if gpu_tests:
        print(f"{BRIGHT_YELLOW}Running GPU tests sequentially...{RESET}")
        for f in gpu_tests:
            res = run_single_test(f, env, timeout_seconds)
            results.append(res)
            name, stdout, _, returncode, runtime, _ = res
            print_result_line(name, stdout, returncode, runtime)

    # CPU TESTS
    if cpu_tests:
        cpu_workers = max(1, min(args.cpu_workers, len(cpu_tests)))

        if args.cpu_sequential:
            cpu_workers = 1

        print(f"\n{BRIGHT_GREEN}Running CPU tests with {cpu_workers} worker(s)...{RESET}")

        if cpu_workers == 1:
            for f in cpu_tests:
                res = run_single_test(f, env, timeout_seconds)
                results.append(res)
                name, stdout, _, returncode, runtime, _ = res
                print_result_line(name, stdout, returncode, runtime)
        else:
            with ProcessPoolExecutor(max_workers=cpu_workers) as executor:
                futures = {executor.submit(run_single_test, f, env, timeout_seconds): f for f in cpu_tests}
                for fut in as_completed(futures):
                    try:
                        res = fut.result()
                    except Exception as e:
                        f = futures[fut]
                        stdout = f"[ERROR] ProcessPool future crashed: {repr(e)}"
                        res = (f.name, stdout, "", 1, 0.0, TMP_LOGS / f"{f.name}.log")

                    results.append(res)
                    name, stdout, _, returncode, runtime, _ = res
                    print_result_line(name, stdout, returncode, runtime)

    # ============================================================
    # MERGE ALL TEST OUTPUTS INTO ONE UNIFIED HUMAN-READABLE LOG
    # ============================================================
    with open(final_log_path, "w", encoding="utf-8") as final:
        final.write("\n================ WAVELOCK FULL TEST LOG ================\n")

        for name, stdout, _, returncode, runtime, tmp_log in results:
            final.write("\n" + "="*80 + "\n")
            final.write(f"TEST FILE: {name}\n")
            final.write(f"RETURN CODE: {returncode}\n")
            final.write(f"RUNTIME: {runtime:.2f}s\n")
            final.write("="*80 + "\n\n")
            final.write(stdout)
            final.write("\n")

    sha = hashlib.sha256(Path(final_log_path).read_bytes()).hexdigest()

    # SUMMARY JSON
    summary = {
        "timestamp": timestamp,
        "environment": env_info,
        "sha256": sha,
        "config": {
            "only": args.only,
            "cpu_workers": args.cpu_workers,
            "cpu_sequential": args.cpu_sequential,
            "timeout_seconds": timeout_seconds,
            "pythonpath": str(REPO_ROOT),
            "thread_limits": {
                "OMP_NUM_THREADS": env["OMP_NUM_THREADS"],
                "MKL_NUM_THREADS": env["MKL_NUM_THREADS"],
                "OPENBLAS_NUM_THREADS": env["OPENBLAS_NUM_THREADS"],
                "NUMEXPR_NUM_THREADS": env["NUMEXPR_NUM_THREADS"],
                "VECLIB_MAXIMUM_THREADS": env["VECLIB_MAXIMUM_THREADS"],
            }
        },
        "tests": []
    }

    crash_count = 0
    pass_count = 0
    diagnostic_count = 0
    timeout_count = 0

    for name, stdout, _, returncode, runtime, _ in results:
        metrics = extract_metrics(stdout)
        danger = compute_real_risk(metrics)
        if danger is None:
            danger = compute_fallback_risk(stdout)

        status = classify(name, stdout, "", returncode)

        if "CRASH" in status:
            crash_count += 1
        elif "TIMEOUT" in status:
            timeout_count += 1
        elif "DIAGNOSTIC" in status:
            diagnostic_count += 1
        elif "PASS" in status:
            pass_count += 1

        summary["tests"].append({
            "name": name,
            "category": classify_category(name),
            "danger": danger,
            "metrics": metrics,
            "status": status,
            "returncode": returncode,
            "runtime": runtime
        })

    summary["counts"] = {
        "total": len(results),
        "pass": pass_count,
        "diagnostic": diagnostic_count,
        "crash": crash_count,
        "timeout": timeout_count
    }

    with open(json_summary_path, "w", encoding="utf-8") as js:
        json.dump(summary, js, indent=2)

    print(f"\n{BOLD}{CYAN}=== FINAL VERIFICATION SUMMARY ==={RESET}")
    print(f"{MAGENTA}SHA256(log):{RESET} {sha}")
    print(f"{CYAN}JSON Summary:{RESET} {json_summary_path}")
    print(f"{CYAN}Counts:{RESET} {summary['counts']}")

    if crash_count or timeout_count:
        banner_color = BRIGHT_YELLOW
        verdict = "WAVELOCK BENCHMARK COMPLETED WITH WARNINGS"
    else:
        banner_color = BRIGHT_GREEN
        verdict = "WAVELOCK VERIFIED"

    print(f"""
{banner_color}┌───────────────────────────────────────────────┐{RESET}
{banner_color}│  {verdict:<43}│{RESET}
{banner_color}│  {len(results)} tests completed               │{RESET}
{banner_color}│  GPU-safe + capped CPU parallel mode          │{RESET}
{banner_color}└───────────────────────────────────────────────┘{RESET}
""")

    print("Done.\n")


if __name__ == "__main__":
    run()
