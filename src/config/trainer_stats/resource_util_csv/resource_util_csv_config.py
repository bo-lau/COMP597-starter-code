from src.config.util.base_config import _Arg, _BaseConfig

config_name = "resource_util_csv"


class TrainerStatsConfig(_BaseConfig):
    """Configuration for sham-bolic-style resource utilization CSV tracker.

    Writes resource_util.csv and resource_util_substeps.csv for use with
    scripts/plotting/plot_resources.py.
    """

    def __init__(self) -> None:
        super().__init__()
        self._arg_output_dir = _Arg(
            type=str,
            help="Directory where resource utilization CSV files will be saved.",
            default=".",
        )
        self._arg_output_file = _Arg(
            type=str,
            help="Output CSV filename (within output_dir).",
            default="resource_util.csv",
        )
        self._arg_substep_output_file = _Arg(
            type=str,
            help="Output CSV filename for substep (phase) data.",
            default="resource_util_substeps.csv",
        )
