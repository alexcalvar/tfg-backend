import os

from src.core.processing_strategies.batch_strategy import BatchStrategy
from src.core.processing_strategies.temporal_strategy import TemporalStrategy
from src.utils.config_loader import ConfigLoader
from src.utils.file_utils import load_json 


class ProcessingFactory:
    
    def __init__(self):
        self.config = ConfigLoader()
        config_folder_path = self.config.get_path("config_folder")
        models_config_path = os.path.join(config_folder_path, "models_config.json")
        self.models_config = load_json(models_config_path)

    def create_strategy(self, strategy_selected):

        match strategy_selected:
            
            case "batch_strategy":
                return BatchStrategy()
            
            case "temporal_strategy":
                return TemporalStrategy()
            
            case _:
                raise ValueError(f"[ERROR] Estrategia de procesamiento no soportada")
            
