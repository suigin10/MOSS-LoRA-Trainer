"""Optional device-wide NVIDIA memory monitoring; no backend imports in the GUI."""
import csv
import io
import math
import os
import subprocess
import threading


def read_gpu_memory():
    """Return index, used and total MiB. Shared system memory is not included."""
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.used,memory.total',
         '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=2, check=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    rows = []
    for row in csv.reader(io.StringIO(result.stdout)):
        if len(row) != 3:
            continue
        try:
            index = int(row[0])
            used, total = float(row[1]), float(row[2])
            if not all(math.isfinite(v) for v in (used, total)) or not 0 <= used <= total or total <= 0:
                continue
            rows.append((index, used, total))
        except ValueError:
            continue
    return rows


class GpuMonitor:
    def __init__(self):
        self.lock = threading.Lock()
        self.cancel = threading.Event()
        self.rows = []
        self.peaks = {}

    def start(self):
        self.stop()
        cancel = self.cancel = threading.Event()
        with self.lock:
            self.rows, self.peaks = [], {}
        threading.Thread(target=self._run, args=(cancel,), daemon=True).start()

    def _run(self, cancel):
        while not cancel.is_set():
            try:
                rows = read_gpu_memory()
            except (OSError, subprocess.SubprocessError, ValueError):
                rows = []
            with self.lock:
                if cancel.is_set():
                    return
                self.rows = rows
                for index, used, total in rows:
                    self.peaks[index] = max(self.peaks.get(index, 0), used)
            if cancel.wait(3):
                return

    def snapshot(self):
        with self.lock:
            return [(i, used, total, self.peaks[i]) for i, used, total in self.rows]

    def stop(self):
        self.cancel.set()
