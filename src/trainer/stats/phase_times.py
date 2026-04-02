"""Per-step phase timings (forward / backward / optimizer) via ``SimpleTrainerStats`` timers."""

from __future__ import annotations

import csv
import logging
import os

import torch

import src.config as config
import src.trainer.stats.base as base
import src.trainer.stats.simple as simple

logger = logging.getLogger(__name__)

trainer_stats_name = "phase_times"


def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    if "device" in kwargs:
        device = kwargs["device"]
    else:
        logger.warning("No device provided to phase_times trainer stats. Using default PyTorch device")
        device = torch.get_default_device()

    output_dir = "."
    output_file = "phase_times.csv"
    pt = getattr(conf.trainer_stats_configs, "phase_times", None)
    if pt is not None:
        output_dir = getattr(pt, "output_dir", ".")
        output_file = getattr(pt, "output_file", "phase_times.csv")

    csv_path = os.path.join(output_dir, output_file)
    return PhaseTimesTrainerStats(device=device, csv_path=csv_path)


class PhaseTimesTrainerStats(simple.SimpleTrainerStats):
    """CUDA-synchronized timers per phase; writes ``phase_times.csv`` at end of training.

    Uses the same timing hooks as ``SimpleTrainerStats`` (sync before/after each phase).
    Does not record GPU/CPU utilization—use ``resource_util`` for that in a separate run
    if needed.
    """

    def __init__(self, device: torch.device, csv_path: str = "phase_times.csv") -> None:
        super().__init__(device)
        self._csv_path = csv_path

    def start_train(self) -> None:
        super().start_train()
        out_dir = os.path.dirname(self._csv_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    def log_step(self) -> None:
        pass

    def log_stats(self) -> None:
        fwd = self.forward_stats.stat.history
        bwd = self.backward_stats.stat.history
        opt = self.optimizer_step_stats.stat.history
        n_steps = min(len(fwd), len(bwd), len(opt))
        if n_steps > 0:
            with open(self._csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "forward_ms", "backward_ms", "optimizer_ms"])
                for i in range(n_steps):
                    writer.writerow(
                        [
                            i + 1,
                            fwd[i] / 1e6 if i < len(fwd) else 0.0,
                            bwd[i] / 1e6 if i < len(bwd) else 0.0,
                            opt[i] / 1e6 if i < len(opt) else 0.0,
                        ]
                    )
            logger.info(f"Phase times saved to {self._csv_path}")
            print(
                f"Phase times (mean ms): forward {self.forward_stats.get_average() / 1e6:.3f} | "
                f"backward {self.backward_stats.get_average() / 1e6:.3f} | "
                f"optimizer {self.optimizer_step_stats.get_average() / 1e6:.3f}"
            )
        else:
            logger.warning("No phase timing rows to write (empty histories).")
