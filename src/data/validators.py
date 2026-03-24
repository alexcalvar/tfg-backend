from pydantic import BaseModel
from dataclasses import dataclass

# transferecnia de frame+ id_frame para message_builders
@dataclass
class VideoFrame:
    frame_id: int
    img_b64: str

#comunicacion entre pipeline y image_processor para generar la lista de frames por peticion
@dataclass
class FramesPath:
    frame_id: int
    frame_path: str
    intentos: int

class FrameResults(BaseModel):
    detectado: bool
    descripcion: str
    frame_id:int

class EventInterval(BaseModel):
    event_id: int
    start_frame: int      # El ID del primer frame donde se detecta el evento
    end_frame: int        # El ID del último frame del evento
    start_timestamp: float # Segundos exactos en el vídeo (start_frame * intervalo)
    end_timestamp: float   # Segundos exactos en el vídeo (end_frame * intervalo)
    #descripcion: str
 

#clases pydantic para gestionar los benchmarks
class GroundTruthFrame(BaseModel):
    frame_id: str         
    is_positive: bool     

class FrameEvaluation(BaseModel):
    frame_id: str
    ground_truth_detectado: bool  
    modelo_detectado: bool       
