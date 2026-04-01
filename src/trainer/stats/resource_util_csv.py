from __future__ import annotations

import csv
import logging
import os
from typing import Optional

import psutil
import pynvml
import torch

import src.config as config
import src.trainer.stats.base as base
import src.trainer.stats.simple as simple

logger = logging.getLogger(__name__)

trainer_stats_name = "resource_util_csv"


def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    if "device" in kwargs:
        device = kwargs["device"]
    else:
        logger.warning("No device provided to resource_util_csv. Using default PyTorch device")
        device = torch.get_default_device()

    output_dir = "."
    output_file = "resource_util.csv"
    substep_output_file = "resource_util_substeps.csv"

    ru_config = getattr(conf.trainer_stats_configs, "resource_util_csv", None)
    if ru_config is not None:
        output_dir = getattr(ru_config, "output_dir", ".")
        output_file = getattr(ru_config, "output_file", "resource_util.csv")
        substep_output_file = getattr(ru_config, "substep_output_file", "resource_util_substeps.csv")

    csv_path = os.path.join(output_dir, output_file)
    substep_csv_path = os.path.join(output_dir, substep_output_file)

    return ResourceUtilCSVStats(device=device, csv_path=csv_path, substep_csv_path=substep_csv_path)


class ResourceUtilCSVStats(simple.SimpleTrainerStats):
    """Records per-step and per-phase resource metrics, writes sham-bolic-format CSVs."""

    def __init__(self, device: torch.device, csv_path: str = "resource_util.csv", substep_csv_path: Optional[str] = None):
        super().__init__(device)
        self.csv_path = csv_path
        self.substep_csv_path = substep_csv_path or csv_path.replace(".csv", "_substeps.csv")
        self._rows = []
        self._substep_rows = []

        self.gpu_handle = None
        if device.type == "cuda":
            pynvml.nvmlInit()
            gpu_index = device.index if getattr(device, "index", None) is not None else 0
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self.process = psutil.Process()
        # CPU utilization (see course announcement on psutil):
        #   - psutil.cpu_percent() = system-wide average over all cores (0–100%).
        #   - psutil.Process().cpu_percent() = this process only, sum across cores (can exceed 100%).
        # We use Process (Option 2): reflects training process CPU; DataLoader workers are separate
        # processes and are not included. State this in your report.

        self.gpu_util_stats = _RunningFloat()
        self.gpu_mem_gb_stats = _RunningFloat()
        self.cpu_util_stats = _RunningFloat()
        self.cpu_mem_gb_stats = _RunningFloat()
        self.ram_gb_stats = _RunningFloat()
        self.io_read_start = 0
        self.io_write_start = 0

    def start_train(self) -> None:
        super().start_train()
        io = self.process.io_counters()
        self.io_read_start = io.read_bytes
        self.io_write_start = io.write_bytes
        self._rows = []
        self._substep_rows = []
        # Prime CPU % so the first training step is not stuck at 0 (psutil baseline).
        self.process.cpu_percent(interval=None)
        logger.info(
            "cpu_util column = Process.cpu_percent() (per-process sum over cores; can exceed 100%). "
            "Not psutil.cpu_percent() (system-wide average)."
        )
        output_dir = os.path.dirname(self.csv_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def _sample_cpu_util(self) -> float:
        return float(self.process.cpu_percent(interval=None))

    def _record_substep(self, phase: str) -> None:
        """Record resource stats at end of a phase (forward/backward/optimizer)."""
        if self.device.type != "cuda" or self.gpu_handle is None:
            return
        torch.cuda.synchronize(self.device)
        step_num = int(self.step_stats.stat.average.n) + 1
        util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
        gpu_util = float(util.gpu)
        gpu_mem_gb = mem_info.used / 1e9
        cpu_util = self._sample_cpu_util()
        cpu_mem_gb = self.process.memory_info().rss / 1e9
        ram_gb = psutil.virtual_memory().used / 1e9
        io = self.process.io_counters()
        io_read_gb = (io.read_bytes - self.io_read_start) / 1e9
        io_write_gb = (io.write_bytes - self.io_write_start) / 1e9
        self._substep_rows.append([
            step_num,
            phase,
            gpu_util,
            cpu_util,
            gpu_mem_gb,
            cpu_mem_gb,
            ram_gb,
            io_read_gb,
            io_write_gb,
        ])

    def start_step(self) -> None:
        super().start_step()

    def stop_step(self) -> None:
        super().stop_step()
        step_num = int(self.step_stats.stat.average.n)
        if self.gpu_handle is not None:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            self.gpu_util_stats.update(float(util.gpu))
            self.gpu_mem_gb_stats.update(mem_info.used / 1e9)
        else:
            self.gpu_util_stats.update(0.0)
            self.gpu_mem_gb_stats.update(0.0)
        self.cpu_util_stats.update(self._sample_cpu_util())
        self.cpu_mem_gb_stats.update(self.process.memory_info().rss / 1e9)
        self.ram_gb_stats.update(psutil.virtual_memory().used / 1e9)

        io = self.process.io_counters()
        io_read_gb = (io.read_bytes - self.io_read_start) / 1e9
        io_write_gb = (io.write_bytes - self.io_write_start) / 1e9

        self._rows.append([
            step_num,
            self.gpu_util_stats.get_last(),
            self.cpu_util_stats.get_last(),
            self.gpu_mem_gb_stats.get_last(),
            self.cpu_mem_gb_stats.get_last(),
            self.ram_gb_stats.get_last(),
            io_read_gb,
            io_write_gb,
        ])

    def start_forward(self) -> None:
        super().start_forward()

    def stop_forward(self) -> None:
        self._record_substep("forward")
        super().stop_forward()

    def start_backward(self) -> None:
        super().start_backward()

    def stop_backward(self) -> None:
        self._record_substep("backward")
        super().stop_backward()

    def start_optimizer_step(self) -> None:
        super().start_optimizer_step()

    def stop_optimizer_step(self) -> None:
        self._record_substep("optimizer")
        super().stop_optimizer_step()

    def stop_train(self) -> None:
        super().stop_train()

    def log_step(self) -> None:
        pass

    def log_stats(self) -> None:
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            # cpu_util: Process.cpu_percent() — per-process sum % (all cores used by this process).
            writer.writerow([
                "step", "gpu_util", "cpu_util", "gpu_mem_gb", "cpu_mem_gb", "ram_gb",
                "io_read_gb", "io_write_gb",
            ])
            writer.writerows(self._rows)
        logger.info(f"Resource utilization saved to {self.csv_path}")

        if self._substep_rows:
            with open(self.substep_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "step", "phase", "gpu_util", "cpu_util", "gpu_mem_gb", "cpu_mem_gb", "ram_gb",
                    "io_read_gb", "io_write_gb",
                ])
                writer.writerows(self._substep_rows)
            logger.info(f"Resource utilization substeps saved to {self.substep_csv_path}")

        # Phase times (ms) for bar chart: mean ± std per phase
        phase_times_path = os.path.join(
            os.path.dirname(self.csv_path),
            "phase_times.csv",
        )
        fwd = self.forward_stats.stat.history
        bwd = self.backward_stats.stat.history
        opt = self.optimizer_step_stats.stat.history
        n_steps = min(len(fwd), len(bwd), len(opt))
        if n_steps > 0:
            with open(phase_times_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "forward_ms", "backward_ms", "optimizer_ms"])
                for i in range(n_steps):
                    writer.writerow([
                        i + 1,
                        fwd[i] / 1e6 if i < len(fwd) else 0,
                        bwd[i] / 1e6 if i < len(bwd) else 0,
                        opt[i] / 1e6 if i < len(opt) else 0,
                    ])
            logger.info(f"Phase times saved to {phase_times_path}")

    def log_loss(self, loss: torch.Tensor) -> None:
        pass


class _RunningFloat:
    """Minimal running value tracker for floats (last value only for CSV rows)."""

    def __init__(self) -> None:
        self._last = 0.0

    def update(self, value: float) -> None:
        self._last = value

    def get_last(self) -> float:
        return self._last
