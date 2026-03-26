from abc import ABC, abstractmethod

from src.data.validators import FrameResults, FramesPath

class BaseFrameParser(ABC):
    @abstractmethod
    def get_format_instructions(self) -> str:
        """Devuelve el fragmento de texto que se inyectará en el prompt con indicaciones
        especificas del tipo parser utilizado"""
        pass

    @abstractmethod
    def parse(self, text: str, frame_id: int) -> FrameResults:
        """Convierte el string en el objeto FrameResults. Enfocado para cuando se realice una peticion 
        y se espere una unica respuesta """
        pass

    @abstractmethod
    def parse_batch(self, text: str, lote: list[FramesPath]) -> list[FrameResults]:
        """
        Lee el texto crudo, lo divide, y devuelve UNA LISTA de objetos FrameResults. Orientado a 
        modod de procesamiento donde se esperen respuestas multiples por peticion
        """
        pass