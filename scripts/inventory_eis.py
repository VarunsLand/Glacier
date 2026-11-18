from ftplib import FTP
from collections import deque
from pathlib import Path
import csv, time

HOST = "sidads.colorado.edu"
BASE = "/pub/DATASETS/nsidc0783_extreme_ice_survey_v1"

OUT = Path.home() / "Desktop" / "GlacierResearch" / "datasets" / "eis"
OUT.mkdir(parents=True, exist_ok=True)

manifest = OUT / "eis_inventory.csv"
errors = OUT / "inventory_errors.txt"

ftp = FTP(HOST, timeout=120)
ftp.login("anonymous", "student@example.com")
ftp.set_pasv(True)

queue = deque([BASE])
total_files = 0
total_bytes = 0
directory_count = 0
failed = []

print("Connected to official NSIDC EIS archive.")
print("Starting recursive inventory...")

with manifest.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["remote_path", "size_bytes"])

    while queue:
        folder = queue.popleft()
        directory_count += 1

        try:
            entries = ftp.mlsd(folder)

            for name, facts in entries:
                if name in (".", ".."):
                    continue

                remote = folder.rstrip("/") + "/" + name
                kind = facts.get("type", "")

                if kind == "dir":
                    queue.append(remote)

                elif kind == "file":
                    size = int(facts.get("size", 0))
                    writer.writerow([remote, size])
                    total_files += 1
                    total_bytes += size

        except Exception as exc:
            failed.append(f"{folder}: {exc}")

        if directory_count % 25 == 0:
            f.flush()
            print(
                f"Directories: {directory_count:,} | "
                f"Files: {total_files:,} | "
                f"Total size: {total_bytes / 1024**3:.2f} GiB"
            )

            time.sleep(0.2)

ftp.quit()

errors.write_text("\n".join(failed), encoding="utf-8")

print("\n========== EIS INVENTORY COMPLETE ==========")
print("Directories:", directory_count)
print("Files:", total_files)
print("Total size: %.2f GiB" % (total_bytes / 1024**3))
print("Manifest:", manifest)
print("Errors:", len(failed))
