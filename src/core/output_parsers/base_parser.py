from abc import ABC, abstractmethod

from src.data.validators import FrameResults, FramesPath

class BaseFrameParser(ABC):
    @abstractmethod
    def get_format_instructions(self) -> str:
        """Devuelve el fragmento de texto que se inyectará en el prompt."""
        pass

    @abstractmethod
    def parse(self, text: str, frame_id: int) -> FrameResults:
        """Convierte el string crudo en el objeto FrameResults."""
        pass

    @abstractmethod
    def parse_batch(self, text: str, lote: list[FramesPath]) -> list[FrameResults]:
        """
        Se usa en BatchStrategy.
        Lee el texto crudo,
        lo divide, y devuelve UNA LISTA de objetos FrameResults.
        """
        pass