from src.data.enums import ParserType, StrategyType

from src.core.processing_strategies.batch_strategy import BatchStrategy
from src.core.processing_strategies.temporal_strategy import TemporalStrategy
from src.core.processing_strategies.base_strategy import ProcessingStrategy

from src.core.output_parsers.json_parser import JsonFrameParser
from src.core.output_parsers.yes_no_parser import YesNoTextParser
from src.core.output_parsers.base_parser import BaseFrameParser

from src.utils.config_loader import ConfigLoader

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ProcessingFactory:
    
    def __init__(self):
        self.config = ConfigLoader()

    
    def create_strategy(self, strategy_selected : str) -> ProcessingStrategy:
        """ Método encargado de crear la instancia del tipo de procesamiento"""
        try:
            strategy_selected = StrategyType(strategy_selected)
        except ValueError:
            logger.exception(f"Estrategia de procesamiento no soportada : {strategy_selected}")
            raise ValueError(f"Estrategia de procesamiento no soportada: {strategy_selected}")

        parser = self._create_parser()
        
        match strategy_selected:
            
            case StrategyType.BATCH:
                return BatchStrategy(parser)
            
            case StrategyType.TEMPORAL:
                return TemporalStrategy(parser)
            
            case _:
                logger.exception(f"Estrategia de procesamiento {strategy_selected} no soportada ")
                raise ValueError(f"[ERROR] Estrategia de procesamiento {strategy_selected} no soportada ")
                
            

    def _create_parser(self) -> BaseFrameParser:
        """Método para crear el parser según la configuración del propertiesj"""
        parser_str = self.config.get_sys_config("output_parser_type")
        
        # string a enum
        try:
            parser_selected = ParserType(parser_str)
        except ValueError:
            logger.exception(f" Parser '{parser_str}' no reconocido en config. Usando JSON por defecto.")
            parser_selected = ParserType.JSON
        
        match parser_selected:
            case ParserType.TEXT:
                logger.info("Construyendo sistema con Parser de Texto Libre (Sí/No)")
                return YesNoTextParser()
            
            case ParserType.JSON:
                logger.info("Construyendo sistema con Parser JSON Estricto")
                return JsonFrameParser()