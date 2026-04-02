from src.config.util.base_config import _Arg, _BaseConfig

config_name = "phase_times"


class TrainerStatsConfig(_BaseConfig):
    """Output path for per-step phase timings (forward / backward / optimizer)."""

    def __init__(self) -> None:
        super().__init__()
        self._arg_output_dir = _Arg(
            type=str,
            help="Directory for phase_times.csv.",
            default=".",
        )
        self._arg_output_file = _Arg(
            type=str,
            help="CSV filename (within output_dir).",
            default="phase_times.csv",
        )
