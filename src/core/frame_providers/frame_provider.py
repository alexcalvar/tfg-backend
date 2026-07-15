from abc import ABC, abstractmethod
import asyncio

class BaseFrameProvider(ABC):

    @abstractmethod
    async def extract_frames(self, cola_frames: asyncio.Queue, cancel_event: asyncio.Event = None) -> None:
        pass

    @abstractmethod
    def get_expected_frame_count(self) -> int:
        """Calcula cuántos frames se extraerán estimativamente antes de iniciar el proceso."""
        pass

    @abstractmethod
    def get_source_name(self) -> str:
        """Devuelve el nombre del origen de datos (nombre del archivo o de la carpeta)."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Libera o elimina los recursos físicos utilizados por el proveedor una vez finalizada la extracción."""
        pass