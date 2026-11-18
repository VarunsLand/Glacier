"""Test launching Streamlit headless server and verify HTTP 200 response."""
import subprocess
import sys
import time
import urllib.request

cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true", "--server.port", "8502"]
proc = subprocess.Popen(cmd, cwd="C:/Users/Varun's LOQ/Desktop/GlacierResearch", stdout=subprocess.PIPE, stderr=subprocess.PIPE)

try:
    print("Waiting for Streamlit server to initialize on port 8502...")
    time.sleep(6)
    
    url = "http://localhost:8502"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=5) as response:
        status = response.status
        content_type = response.headers.get_content_type()
        print(f"HTTP Status: {status} ({content_type})")
        if status == 200:
            print("[PASS] Streamlit web application started and responded successfully!")
        else:
            print(f"[FAIL] Unexpected status code: {status}")
            sys.exit(1)
finally:
    proc.terminate()
    proc.wait()
    print("Streamlit test server shut down cleanly.")
