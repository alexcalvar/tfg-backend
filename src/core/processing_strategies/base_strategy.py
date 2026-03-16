from abc import ABC, abstractmethod
import asyncio

from core.image_processor import VLMProcessor
from utils.config_loader import ConfigLoader


class ProcessingStrategy(ABC):
    """
    interfaz comun a todos los tipos de procesamiento de frames 
    """
    def __init__(self):
        self.config = ConfigLoader()

    @abstractmethod
    def load_prompts(self) -> tuple[str, str]:
        """Metodo para leer de la configuracion el system promprt del tipo de procesamiento concreto"""
        pass

    @abstractmethod
    async def procesar_lote(self, processor: VLMProcessor, prompt_usuario: str, lote: list, cola: asyncio.Queue, 
                            resultados: list):
        """Metodo para enviar al modelo los frames para procesar"""
        pass