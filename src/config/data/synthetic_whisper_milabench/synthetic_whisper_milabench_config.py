from src.config.util.base_config import _Arg, _BaseConfig

config_name = "synthetic_whisper_milabench"


class DataConfig(_BaseConfig):
    """Config for Milabench-style synthetic Whisper data (in-memory, repeat).

    Unique sample count is ``--batch_size`` (not configured here), matching sham-bolic
    ``synthetic_whisper`` memory mode.
    """

    def __init__(self) -> None:
        super().__init__()
        self._arg_repeat = _Arg(
            type=int,
            help="Repeat multiplier. Effective dataset size = batch_size * repeat.",
            default=10,
        )
        self._arg_num_labels = _Arg(
            type=int,
            help="Number of classification labels.",
            default=10,
        )
