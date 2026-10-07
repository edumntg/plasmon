"""Hardware facts and live metrics for registration and heartbeats."""

from __future__ import annotations

import platform
from typing import Any

import psutil


def _gpu() -> dict[str, Any]:
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {"kind": "cuda", "name": props.name, "vram_gb": round(props.total_memory / 2**30, 1), "count": torch.cuda.device_count()}
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return {"kind": "mps", "name": "Apple GPU", "vram_gb": round(psutil.virtual_memory().total / 2**30, 1), "count": 1}
    except Exception:  # torch missing or broken: report no GPU
        pass
    return {"kind": "none"}


def benchmark_tflops(seconds: float = 0.3) -> float | None:
    """Matmul throughput of the device the trainer would use, in TFLOPS. A fixed time budget
    keeps it short on a CPU; fp16 on CUDA, where tensor cores do the training work."""
    try:
        import time

        import torch

        if torch.cuda.is_available():
            device, dtype, n = torch.device("cuda"), torch.float16, 4096
        else:
            device, dtype, n = torch.device("cpu"), torch.float32, 1024
        a = torch.randn(n, n, device=device, dtype=dtype)
        b = torch.randn(n, n, device=device, dtype=dtype)
        a @ b  # warm up kernels and allocator
        if device.type == "cuda":
            torch.cuda.synchronize()
        flops_per_matmul = 2.0 * n * n * n
        count = 0
        start = time.perf_counter()
        while time.perf_counter() - start < seconds:
            a @ b
            count += 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        return round(count * flops_per_matmul / elapsed / 1e12, 3)
    except Exception:
        return None


def hardware() -> dict[str, Any]:
    out = {
        "hostname": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "arch": platform.machine(),
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "gpu": _gpu(),
    }
    tflops = benchmark_tflops()
    if tflops is not None:
        out["tflops"] = tflops
    return out


def versions() -> dict[str, str]:
    from .. import __version__

    out = {"plasmon": __version__, "python": platform.python_version()}
    try:
        import torch

        out["torch"] = torch.__version__
    except Exception:
        pass
    return out


class _Nvml:
    def __init__(self):
        self.handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self.pynvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self.pynvml = None

    def read(self) -> dict[str, Any]:
        if self.handle is None:
            return {}
        try:
            util = self.pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            mem = self.pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            out = {"gpu_pct": util.gpu, "vram_used_gb": round(mem.used / 2**30, 2), "vram_total_gb": round(mem.total / 2**30, 2)}
            try:
                out["gpu_temp_c"] = self.pynvml.nvmlDeviceGetTemperature(self.handle, 0)
            except Exception:
                pass
            return out
        except Exception:
            return {}


_nvml = _Nvml()
_last_net = None


def metrics() -> dict[str, Any]:
    global _last_net
    vm = psutil.virtual_memory()
    out: dict[str, Any] = {"cpu_pct": psutil.cpu_percent(interval=None), "ram_pct": vm.percent, "ram_used_gb": round(vm.used / 2**30, 2)}
    try:
        batt = psutil.sensors_battery()
        if batt is not None:
            out["on_battery"] = not batt.power_plugged
            out["battery_pct"] = batt.percent
    except Exception:
        pass
    try:
        net = psutil.net_io_counters()
        import time

        now = time.time()
        if _last_net:
            dt_ = max(now - _last_net[0], 1e-3)
            out["net_up_mbps"] = round(8 * (net.bytes_sent - _last_net[1]) / dt_ / 1e6, 2)
            out["net_down_mbps"] = round(8 * (net.bytes_recv - _last_net[2]) / dt_ / 1e6, 2)
        _last_net = (now, net.bytes_sent, net.bytes_recv)
    except Exception:
        pass
    out.update(_nvml.read())
    return out
