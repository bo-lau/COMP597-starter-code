from src.config.util.base_config import _Arg, _BaseConfig

config_name = "resource_util_max"


class TrainerStatsConfig(_BaseConfig):
    """Configuration for ``resource_util_max`` (per-step CSV + train duration file)."""

    def __init__(self) -> None:
        super().__init__()
        self._arg_output_dir = _Arg(
            type=str,
            help="Output directory for resource_util.csv and duration txt.",
            default=".",
        )
