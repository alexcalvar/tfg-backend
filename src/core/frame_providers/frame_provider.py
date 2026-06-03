from abc import ABC,abstractmethod
import asyncio

class BaseFrameProvider(ABC):

    @abstractmethod
    async def extract_frames(self, cola_frames: asyncio.Queue) -> None:
        """ Extrae de forma asíncrona las rutas y metadatos de los fotogramas y los encola para su procesamiento"""
        pass

    @abstractmethod
    def get_expected_frame_count(self) -> int:
        pass

    @abstractmethod
    def get_source_name(self) -> str:
        """Devuelve el nombre del origen de datos (nombre del archivo o de la carpeta)"""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Libera o elimina los recursos físicos utilizados por el proveedor una vez finalizada la extracción."""
    pass