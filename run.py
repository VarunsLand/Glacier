"""GlacierVision One-Click Application Launcher.

Cross-platform runner for the GlacierVision AI Cryosphere Monitoring Platform.
Automatically detects environment, checks port availability, opens browser, and runs Streamlit.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path


def is_server_healthy(port: int = 8501) -> bool:
    """Check if a Streamlit instance is already responding on the port."""
    try:
        url = f"http://localhost:{port}/_stcore/health"
        req = urllib.request.Request(url, headers={"User-Agent": "GlacierVisionLauncher"})
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        return False


def find_available_port(start_port: int = 8501, max_tries: int = 20) -> int:
    """Find the first available TCP port starting from start_port."""
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start_port


def main() -> None:
    # 1. Set working directory to repository root
    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)

    print("=" * 72)
    print("        ❄️ GLACIERVISION AI CRYOSPHERE MONITORING PLATFORM")
    print("=" * 72)
    print(f"[INFO] Working Directory : {repo_root}")
    print(f"[INFO] Python Executable : {sys.executable} (v{sys.version.split()[0]})")

    # 2. Check if GlacierVision is already running on 8501
    if is_server_healthy(8501):
        url = "http://localhost:8501"
        print("\n" + "=" * 72)
        print("  🎉 GlacierVision is ALREADY RUNNING and active at:")
        print(f"     👉 {url}")
        print("=" * 72)
        print("\n[INFO] Opening your browser to the active session now...")
        try:
            webbrowser.open(url)
        except Exception:
            pass
        print("\nApplication is ready! You can interact with it in your browser.")
        print("If you wish to stop the active server, close the terminal window where it started.")
        print("Press Enter to close this launcher.")
        try:
            input()
        except (KeyboardInterrupt, EOFError):
            pass
        return

    # 3. Find an available port
    target_port = find_available_port(8501)
    url = f"http://localhost:{target_port}"
    print(f"[INFO] Selected Port     : {target_port}")
    print(f"[INFO] Target URL        : {url}")
    print("=" * 72)

    # 4. Schedule browser opening
    def open_browser():
        time.sleep(2)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    threading.Thread(target=open_browser, daemon=True).start()

    # 5. Launch Streamlit server
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        f"--server.port={target_port}",
        "--server.headless=false",
        "--browser.gatherUsageStats=false",
    ]

    print("\n[INFO] Starting Streamlit server... (Press Ctrl+C to stop)\n")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[INFO] GlacierVision server stopped cleanly.")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Streamlit exited with code {e.returncode}")
        print("Press Enter to exit.")
        try:
            input()
        except Exception:
            pass
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
