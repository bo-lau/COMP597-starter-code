from src.config.util.base_config import _Arg, _BaseConfig

config_name = "codecarbon_e2e"


class TrainerStatsConfig(_BaseConfig):
    """CodeCarbon with a long power-sampling interval (default: ~one sample per short run).

    Use this to estimate **end-to-end energy** with **minimal CodeCarbon polling overhead**,
    and compare against ``--trainer_stats codecarbon`` (e.g. ``measure_power_secs 0.5``).
    """

    def __init__(self) -> None:
        super().__init__()
        self._arg_run_num = _Arg(
            type=int,
            help="The run number used for codecarbon file tracking.",
            default=0,
        )
        self._arg_project_name = _Arg(
            type=str,
            help="The name of the project used for codecarbon file tracking.",
            default="energy-efficiency-e2e",
        )
        self._arg_output_dir = _Arg(
            type=str,
            help="The path of the output directory where files will be saved.",
            default=".",
        )
        self._arg_measure_power_secs = _Arg(
            type=float,
            help=(
                "Seconds between CodeCarbon power polls. Default 86400 (24h) is much longer than "
                "typical COMP597 runs, so you get ~one coarse sample over the whole job—use to "
                "assess overhead vs frequent sampling. Override if your run is longer than a day."
            ),
            default=86400.0,
        )
