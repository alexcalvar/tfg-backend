from pydantic import BaseModel

# El modelo para el Dataset 
class GroundTruthFrame(BaseModel):
    frame_id: str         
    is_positive: bool     

# El modelo para el Juez (La fusión para calcular métricas)
class FrameEvaluation(BaseModel):
    frame_id: str
    ground_truth_detectado: bool  # Se saca del GroundTruthFrame
    modelo_detectado: bool        # Se saca de tu JSON crudo de la IA