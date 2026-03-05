from typing import Generic, TypeVar, Optional
from pydantic import BaseModel

# Definimos una variable de tipo 'T' 
T = TypeVar('T')

class HTTPResponse(BaseModel, Generic[T]):
    """Envoltorio base para TODAS las respuestas de la API."""
    success: bool = True
    message: str
    data: Optional[T] = None  # Aquí inyectaremos dinámicamente los datos específicos


class ProjectProgress(BaseModel):
    current_frame: int
    total_frames: int

class StatusData(BaseModel): 
    state: str
    progress: ProjectProgress
    last_updated: str

class AnalyzeData(BaseModel):
    project_id: str
    video_file: str