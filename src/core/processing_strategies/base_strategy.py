from abc import abstractmethod
import asyncio

from src.data.validators import FramesPath
from src.core.image_processor import VLMProcessor
from src.utils.config_loader import ConfigLoader
from src.observer.observer import StatusObservable
from src.core.output_parsers.base_parser import BaseFrameParser

class ProcessingStrategy(StatusObservable):
    """
    Interfaz común a todos los tipos de procesamiento de frames.
    Al heredar de StatusObservable, todas las estrategias tienen el método self.notify().
    """
    
    def __init__(self, parser : BaseFrameParser):
        # inicializa la lista de observadores
        StatusObservable.__init__(self) 
        
        #  inicializa la configuración compartida
        self.config = ConfigLoader()
        self.parser = parser

    @abstractmethod
    def load_prompts(self) -> tuple[str, str]:
        """Método para leer de la configuración el system prompt."""
        pass

    @abstractmethod
    async def process_queue(self, processor: VLMProcessor, prompt_usuario: str, cola: asyncio.Queue, resultados: list) -> None:
        """Consume la cola de frames y procesa según la lógica de la estrategia."""
        pass

    @abstractmethod
    def _build_model_request(self, prompt_usuario : str, lote : list[FramesPath]):
        """ Metodo para crear la peticion que recibira el modelo con el formato correspondiente al tipo de procesamiento"""
        pass