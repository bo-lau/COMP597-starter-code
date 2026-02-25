from src.config.util.base_config import _Arg, _BaseConfig

config_name = "resource_util"

class TrainerStatsConfig(_BaseConfig):
    """Configuration for resource utilization statistics tracker.
    
    Attributes
    ----------
    output_dir : str
        Output directory where resource utilization CSV file will be saved.
    """
    
    def __init__(self) -> None:
        super().__init__()
        self._arg_output_dir = _Arg(
            type=str, 
            help="Output directory for resource utilization CSV file.", 
            default="."
        )
