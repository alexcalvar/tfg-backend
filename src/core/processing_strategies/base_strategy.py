from abc import ABC, abstractmethod
import asyncio

from src.core.image_processor import VLMProcessor
from src.utils.config_loader import ConfigLoader
from src.observer.observer import StatusObservable

class ProcessingStrategy(StatusObservable):
    """
    Interfaz común a todos los tipos de procesamiento de frames.
    Al heredar de StatusObservable, todas las estrategias tienen el método self.notify().
    """
    
    def __init__(self):
        # Inicializa la lista de observadores
        StatusObservable.__init__(self) 
        
        #  Inicializa la configuración compartida
        self.config = ConfigLoader()

    @abstractmethod
    def load_prompts(self) -> tuple[str, str]:
        """Método para leer de la configuración el system prompt."""
        pass

    @abstractmethod
    async def procesar_cola(self, processor: VLMProcessor, prompt_usuario: str, cola: asyncio.Queue, resultados: list):
        """Consume la cola de frames y procesa según la lógica de la estrategia."""
        pass