from src.config.util.base_config import _Arg, _BaseConfig


class DataConfig(_BaseConfig):

    def __init__(self) -> None:
        super().__init__()
        self._arg_data_path = _Arg(
            type=str,
            help="Path to save/load cached synthetic Whisper data.",
            default="synthetic_whisper_data.pt"
        )
        self._arg_num_labels = _Arg(
            type=int,
            help="Number of classification labels for synthetic data.",
            default=10
        )
        self._arg_force_regenerate = _Arg(
            type=int,
            help="If non-zero, regenerate synthetic data even when cache exists.",
            default=0
        )
