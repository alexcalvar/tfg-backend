from src.core.processing_strategies.batch_strategy import BatchStrategy
from src.core.processing_strategies.temporal_strategy import TemporalStrategy
from src.core.processing_strategies.base_strategy import ProcessingStrategy


class ProcessingFactory:

    @staticmethod
    def create_strategy(strategy_selected : str) -> ProcessingStrategy:

        match strategy_selected:
            
            case "batch_strategy":
                return BatchStrategy()
            
            case "temporal_strategy":
                return TemporalStrategy()
            
            case _:
                raise ValueError(f"[ERROR] Estrategia de procesamiento {strategy_selected} no soportada ")
            
