from src.data.enums import ParserType, StrategyType

from src.core.processing_strategies.batch_strategy import BatchStrategy
from src.core.processing_strategies.temporal_strategy import TemporalStrategy
from src.core.processing_strategies.base_strategy import ProcessingStrategy

from src.core.output_parsers.json_parser import JsonFrameParser
from src.core.output_parsers.yes_no_parser import YesNoTextParser
from src.core.output_parsers.base_parser import BaseFrameParser

from src.utils.config_loader import ConfigLoader


class ProcessingFactory:
    
    def __init__(self):
        self.config = ConfigLoader()

    def _create_parser(self) -> BaseFrameParser:
        """Método factoría interno para instanciar el parser según la configuración."""
        parser_str = self.config.get_sys_config("output_parser_type")
        
        # Mapeo seguro del string al Enum
        try:
            parser_selected = ParserType(parser_str)
        except ValueError:
            print(f" [ALERTA] Parser '{parser_str}' no reconocido en config. Usando JSON por defecto.")
            parser_selected = ParserType.JSON
        
        match parser_selected:
            case ParserType.TEXT:
                print(" [INFO] Construyendo sistema con Parser de Texto Libre (Sí/No)")
                return YesNoTextParser()
            
            case ParserType.JSON:
                print(" [INFO] Construyendo sistema con Parser JSON Estricto")
                return JsonFrameParser()

    
    def create_strategy(self, strategy_selected : str) -> ProcessingStrategy:

        try:
            strategy_selected = StrategyType(strategy_selected)
        except ValueError:
            raise ValueError(f"Estrategia de procesamiento no soportada: {strategy_selected}")

        parser = self._create_parser()
        
        match strategy_selected:
            
            case StrategyType.BATCH:
                return BatchStrategy(parser)
            
            case StrategyType.TEMPORAL:
                return TemporalStrategy(parser)
            
            case _:
                raise ValueError(f"[ERROR] Estrategia de procesamiento {strategy_selected} no soportada ")
            
