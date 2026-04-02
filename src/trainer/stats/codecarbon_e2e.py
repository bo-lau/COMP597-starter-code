"""CodeCarbon with minimal power polling (long ``measure_power_secs``) for overhead comparison."""

import logging

import torch

import src.config as config
import src.trainer.stats.base as base
from src.trainer.stats.codecarbon import CodeCarbonStats

logger = logging.getLogger(__name__)

trainer_stats_name = "codecarbon_e2e"

_DEFAULT_MEASURE_POWER_SECS = 86400.0  # 24h — typical runs finish first → ~one sample / minimal overhead


def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    """Same implementation as ``codecarbon`` but defaults to sparse power measurements."""
    if "device" in kwargs:
        device = kwargs["device"]
    else:
        logger.warning("No device provided to codecarbon_e2e trainer stats. Using default PyTorch device")
        device = torch.get_default_device()

    cc = conf.trainer_stats_configs.codecarbon_e2e
    measure = getattr(cc, "measure_power_secs", _DEFAULT_MEASURE_POWER_SECS)
    logger.info(
        "codecarbon_e2e: measure_power_secs=%s (sparse sampling; compare overhead vs --trainer_stats codecarbon)",
        measure,
    )
    return CodeCarbonStats(
        device,
        cc.run_num,
        cc.project_name,
        cc.output_dir,
        measure_power_secs=measure,
    )
