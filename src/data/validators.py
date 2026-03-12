from pydantic import BaseModel
from dataclasses import dataclass

class GroundTruthFrame(BaseModel):
    frame_id: str         
    is_positive: bool     

class FrameEvaluation(BaseModel):
    frame_id: str
    ground_truth_detectado: bool  
    modelo_detectado: bool       


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