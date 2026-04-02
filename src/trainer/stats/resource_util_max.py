"""Per-step resource CSV (sham-bolic columns) + full-loop wall time.

Samples NVML and process CPU at **end of each step** after CUDA sync (snapshot, not
time-averaged like the default ``resource_util`` step stream). Writes
``resource_util.csv`` compatible with ``plot_resources.py``.
"""

import csv
import logging
import os
import time
from pathlib import Path
from typing import Optional

import pynvml
import psutil
import torch

import src.config as config
import src.trainer.stats.base as base
import src.trainer.stats.simple as simple
import src.trainer.stats.utils as utils

logger = logging.getLogger(__name__)

trainer_stats_name = "resource_util_max"


def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    if "device" in kwargs:
        device = kwargs["device"]
    else:
        logger.warning(
            "No device provided to resource_util_max trainer stats. Using default PyTorch device"
        )
        device = torch.get_default_device()

    output_dir = "."
    try:
        if hasattr(conf.trainer_stats_configs, "resource_util_max"):
            output_dir = conf.trainer_stats_configs.resource_util_max.output_dir
    except AttributeError:
        pass

    csv_path = os.path.join(output_dir, "resource_util.csv")
    duration_path = Path(output_dir) / "resource_util_max_train_duration.txt"
    return ResourceUtilMaxStats(
        device=device, csv_path=csv_path, duration_path=duration_path
    )


class ResourceUtilMaxStats(simple.SimpleTrainerStats):
    """Per-step resource CSV plus full-loop wall time (``noop``-style duration).

    Writes ``duration_ms`` to ``resource_util_max_train_duration.txt`` next to the CSV.
    """

    SUPPRESS_PROGRESS_BAR = True

    def __init__(
        self,
        device: torch.device,
        csv_path: str = "resource_util.csv",
        duration_path: Optional[Path] = None,
    ) -> None:
        super().__init__(device)
        self.csv_path = csv_path
        self._duration_path = Path(duration_path) if duration_path is not None else None
        self._rows: list[list] = []
        self._train_start_ns: int | None = None
        self._train_duration_ns: int | None = None

        self.gpu_available = False
        self.gpu_handle = None
        try:
            pynvml.nvmlInit()
            if device.type == "cuda":
                gpu_index = device.index if device.index is not None else 0
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                self.gpu_available = True
        except Exception as e:
            logger.warning("resource_util_max: GPU monitoring unavailable: %s", e)

        self.process = psutil.Process(os.getpid())

        self.gpu_util_stats = utils.RunningStat()
        self.gpu_mem_usage_stats = utils.RunningStat()
        self.cpu_util_stats = utils.RunningStat()
        self.cpu_mem_stats = utils.RunningStat()
        self.ram_usage_stats = utils.RunningStat()

        self.io_read_start = 0
        self.io_write_start = 0
        self.total_io_read = 0
        self.total_io_write = 0

    def start_train(self) -> None:
        self._train_duration_ns = None
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._train_start_ns = time.perf_counter_ns()
        try:
            io = self.process.io_counters()
            self.io_read_start = io.read_bytes
            self.io_write_start = io.write_bytes
        except Exception as e:
            logger.debug("io_counters at start_train: %s", e)
        self.process.cpu_percent(interval=None)
        output_dir = os.path.dirname(self.csv_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        self._rows = []

    def start_step(self) -> None:
        self.step_stats.start()

    def stop_step(self) -> None:
        self.step_stats.stop()
        step_num = int(self.step_stats.stat.average.n)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        if self.gpu_available and self.gpu_handle is not None:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            self.gpu_util_stats.update(int(util.gpu))
            self.gpu_mem_usage_stats.update(int(mem_info.used))
        else:
            self.gpu_util_stats.update(0)
            self.gpu_mem_usage_stats.update(0)

        try:
            cpu_u = float(self.process.cpu_percent(interval=None))
        except Exception:
            cpu_u = 0.0
        self.cpu_util_stats.update(int(round(cpu_u * 100)))
        self.cpu_mem_stats.update(int(self.process.memory_info().rss))
        self.ram_usage_stats.update(int(psutil.virtual_memory().used))

        try:
            io = self.process.io_counters()
            io_read = (io.read_bytes - self.io_read_start) / 1e9
            io_write = (io.write_bytes - self.io_write_start) / 1e9
        except Exception:
            io_read = 0.0
            io_write = 0.0

        self._rows.append(
            [
                step_num,
                self.gpu_util_stats.get_last(),
                self.cpu_util_stats.get_last() / 100.0,
                self.gpu_mem_usage_stats.get_last() / 1e9,
                self.cpu_mem_stats.get_last() / 1e9,
                self.ram_usage_stats.get_last() / 1e9,
                io_read,
                io_write,
            ]
        )

    def stop_train(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        if self._train_start_ns is not None:
            self._train_duration_ns = time.perf_counter_ns() - self._train_start_ns
        self._train_start_ns = None
        try:
            io = self.process.io_counters()
            self.total_io_read = io.read_bytes - self.io_read_start
            self.total_io_write = io.write_bytes - self.io_write_start
        except Exception:
            self.total_io_read = 0
            self.total_io_write = 0

    def log_stats(self) -> None:
        if self._rows:
            gpu_util_mean = sum(r[1] for r in self._rows) / len(self._rows)
            cpu_util_mean = sum(r[2] for r in self._rows) / len(self._rows)
            gpu_mem_mean = sum(r[3] for r in self._rows) / len(self._rows)
            cpu_mem_mean = sum(r[4] for r in self._rows) / len(self._rows)
            ram_mean = sum(r[5] for r in self._rows) / len(self._rows)
            print("###############   RESOURCE UTILIZATION (mean)   ###############")
            print(f"GPU util: {gpu_util_mean:.1f}%   CPU util: {cpu_util_mean:.1f}%")
            print(
                f"GPU mem: {gpu_mem_mean:.4f} GB   CPU mem: {cpu_mem_mean:.4f} GB   RAM: {ram_mean:.4f} GB"
            )
        print("###############   I/O TOTALS   ###############")
        print(f"Total I/O Read: {self.total_io_read / 1e9:.2f} GB")
        print(f"Total I/O Write: {self.total_io_write / 1e9:.2f} GB")
        if self._rows:
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "step",
                        "gpu_util",
                        "cpu_util",
                        "gpu_mem_gb",
                        "cpu_mem_gb",
                        "ram_gb",
                        "io_read_gb",
                        "io_write_gb",
                    ]
                )
                writer.writerows(self._rows)
            logger.info("Resource utilization saved to %s", self.csv_path)
        d = self._train_duration_ns
        path = self._duration_path
        if d is not None and path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            ms = d / 1e6
            with path.open("w", encoding="utf-8") as f:
                f.write(f"duration_ms {ms}\n")
            logger.info("Train duration saved to %s", path)

    def log_step(self) -> None:
        pass
