import asyncio
from typing import Dict, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

class TaskRegistry:
    """
    Registro centralizado (Singleton) para gestionar los eventos de cancelación
    de los proyectos en segundo plano. Previen fugas de memoria y sincroniza HTTP.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TaskRegistry, cls).__new__(cls)
            cls._instance._active_tasks: Dict[str, asyncio.Event] = {}
        return cls._instance

    def register(self, project_id: str) -> asyncio.Event:
        """Crea y registra un nuevo evento de cancelación para un proyecto."""
        event = asyncio.Event()
        self._active_tasks[project_id] = event
        logger.info(f"Proyecto [{project_id}] registrado en el gestor de tareas activas.")
        return event

    def get(self, project_id: str) -> Optional[asyncio.Event]:
        """Recupera el evento de cancelación de un proyecto activo."""
        return self._active_tasks.get(project_id)

    def unregister(self, project_id: str) -> None:
        """Elimina el proyecto del registro para liberar memoria."""
        if project_id in self._active_tasks:
            del self._active_tasks[project_id]
            logger.info(f"Proyecto [{project_id}] eliminado del registro de tareas activas.")