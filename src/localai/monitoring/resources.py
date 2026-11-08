"""System resource monitoring for CPU, RAM, and NVIDIA GPU VRAM telemetry."""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from typing import Optional, Tuple

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SystemResourceSnapshot(BaseModel):
    """Snapshot of current system and GPU memory utilization."""

    cpu_percent: float = 0.0
    ram_used_mb: float = 0.0
    ram_total_mb: float = 0.0
    ram_percent: float = 0.0
    gpu_name: Optional[str] = None
    vram_used_mb: Optional[float] = None
    vram_total_mb: Optional[float] = None
    vram_percent: Optional[float] = None


class ResourceMonitor:
    """Samples and tracks system and GPU resource consumption."""

    def __init__(self) -> None:
        self._tracking = False
        self._track_thread: Optional[threading.Thread] = None
        self._peak_ram_mb: float = 0.0
        self._peak_vram_mb: Optional[float] = None
        self._nvml_initialized = False
        self._gpu_handle = None

        self._init_nvml()

    def _init_nvml(self) -> None:
        """Attempt to initialize pynvml for low-overhead GPU metrics."""
        try:
            import pynvml

            pynvml.nvmlInit()
            if pynvml.nvmlDeviceGetCount() > 0:
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._nvml_initialized = True
        except Exception:
            self._nvml_initialized = False
            self._gpu_handle = None

    def get_snapshot(self) -> SystemResourceSnapshot:
        """Capture current instant resource snapshot."""
        ram_used = 0.0
        ram_total = 0.0
        ram_pct = 0.0
        cpu_pct = 0.0

        try:
            import psutil

            mem = psutil.virtual_memory()
            ram_used = round(mem.used / (1024**2), 1)
            ram_total = round(mem.total / (1024**2), 1)
            ram_pct = round(mem.percent, 1)
            cpu_pct = round(psutil.cpu_percent(interval=None), 1)
        except Exception:
            pass

        gpu_name, vram_used, vram_total, vram_pct = self._query_gpu()

        return SystemResourceSnapshot(
            cpu_percent=cpu_pct,
            ram_used_mb=ram_used,
            ram_total_mb=ram_total,
            ram_percent=ram_pct,
            gpu_name=gpu_name,
            vram_used_mb=vram_used,
            vram_total_mb=vram_total,
            vram_percent=vram_pct,
        )

    def _query_gpu(self) -> Tuple[Optional[str], Optional[float], Optional[float], Optional[float]]:
        """Query GPU memory via NVML or fallback to nvidia-smi."""
        # Try NVML
        if self._nvml_initialized and self._gpu_handle:
            try:
                import pynvml

                name = pynvml.nvmlDeviceGetName(self._gpu_handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8")
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                used_mb = round(mem_info.used / (1024**2), 1)
                total_mb = round(mem_info.total / (1024**2), 1)
                pct = round((used_mb / total_mb) * 100.0, 1) if total_mb > 0 else 0.0
                return name, used_mb, total_mb, pct
            except Exception:
                pass

        # Fallback to nvidia-smi CLI
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=2.0)
            if result.returncode == 0 and result.stdout.strip():
                parts = [p.strip() for p in result.stdout.strip().split(",")]
                if len(parts) >= 3:
                    gpu_name = parts[0]
                    used_mb = float(parts[1])
                    total_mb = float(parts[2])
                    pct = round((used_mb / total_mb) * 100.0, 1) if total_mb > 0 else 0.0
                    return gpu_name, used_mb, total_mb, pct
        except Exception:
            pass

        return None, None, None, None

    def start_tracking(self, interval_sec: float = 0.05) -> None:
        """Begin polling resources in background thread to catch peaks during generation."""
        snapshot = self.get_snapshot()
        self._peak_ram_mb = snapshot.ram_used_mb
        self._peak_vram_mb = snapshot.vram_used_mb
        self._tracking = True

        def _track() -> None:
            while self._tracking:
                snap = self.get_snapshot()
                if snap.ram_used_mb > self._peak_ram_mb:
                    self._peak_ram_mb = snap.ram_used_mb
                if snap.vram_used_mb is not None:
                    if self._peak_vram_mb is None or snap.vram_used_mb > self._peak_vram_mb:
                        self._peak_vram_mb = snap.vram_used_mb
                time.sleep(interval_sec)

        self._track_thread = threading.Thread(target=_track, daemon=True)
        self._track_thread.start()

    def stop_tracking(self) -> Tuple[float, Optional[float]]:
        """Stop background sampling and return (peak_ram_mb, peak_vram_mb)."""
        self._tracking = False
        if self._track_thread and self._track_thread.is_alive():
            self._track_thread.join(timeout=1.0)
        return self._peak_ram_mb, self._peak_vram_mb
