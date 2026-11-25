#!/usr/bin/env python3
"""Run the supported WaveLock-OTS workflow in a disposable local directory.

Each command is a fresh process. Verification runs after deleting secret keys.
This demonstrates public authentication and durable replay rejection.
"""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile


def main():
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="wavelock-ots-demo-") as directory:
        work = Path(directory)
        env = {**os.environ, "PYTHONPATH": str(root),
               "WAVELOCK_DATA_DIR": str(work / "node"),
               "WAVELOCK_OTS_STATE_DIR": str(work / "signer-state"),
               "WAVELOCK_REQUIRE_OTS": "1",
               "POW_TARGET": "0" + "f" * 63}
        env.pop("WAVELOCK_CONFIG", None)
        env.pop("WAVELOCK_OTS_LEDGER", None)
        def run(*arguments, expected=0):
            result = subprocess.run(
                [sys.executable, "-m", "wavelock.chain.cli", *arguments],
                cwd=work, env=env, capture_output=True, text=True, timeout=60,
            )
            print(result.stdout, end="")
            if result.returncode != expected:
                print(result.stderr, file=sys.stderr, end="")
                raise RuntimeError(f"command failed: {arguments}")
            return result
        run("keygen")
        run("sign", "--message", "hello wavelock", "--difficulty", "1")
        run("verify", "--signed-path", "signed_message.json")
        run("mine")
        shutil.rmtree(work / "keys")
        run("verify")
        run("mine", expected=1)
        print("PASS: public-only verification after restart; replay rejected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
