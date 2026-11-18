"""GlacierVision One-Click Application Launcher.

Cross-platform direct runner for the GlacierVision AI Cryosphere Monitoring Platform.
Automatically detects your environment, verifies dependencies, and starts the Streamlit app.

Usage:
    python run.py
    py run.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


def main() -> None:
    # 1. Ensure working directory is always repository root
    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)

    print("=" * 70)
    print("        ❄️ GLACIERVISION AI CRYOSPHERE MONITORING PLATFORM")
    print("=" * 70)
    print(f"[INFO] Working Directory : {repo_root}")
    print(f"[INFO] Python Executable : {sys.executable} (v{sys.version.split()[0]})")

    # 2. Verify Streamlit is installed
    try:
        import streamlit  # noqa: F401
        print("[INFO] Streamlit Status  : Installed")
    except ImportError:
        print("\n[WARNING] Streamlit is not installed in the active Python environment.")
        print("[INFO] Installing required dependencies...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "streamlit", "torch", "torchvision",
                "pandas", "numpy", "matplotlib", "plotly", "geopandas", "shapely", "rasterio", "scikit-learn"
            ])
        except Exception as e:
            print(f"[ERROR] Dependency installation failed: {e}")
            print("Please run: pip install -r requirements.txt (or install streamlit manually).")
            sys.exit(1)

    port = 8501
    url = f"http://localhost:{port}"

    print(f"[INFO] Launching Platform : {url}")
    print("=" * 70)
    print("Press Ctrl+C in this console to stop the application.\n")

    # 3. Schedule browser opening after 2 seconds
    def open_browser():
        time.sleep(2)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    # 4. Run Streamlit application directly
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        f"--server.port={port}",
        "--server.headless=false",
        "--browser.gatherUsageStats=false",
    ]

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[INFO] GlacierVision stopped cleanly.")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Streamlit exited with returncode {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
