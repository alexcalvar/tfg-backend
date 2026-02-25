from pydantic import BaseModel, Field

class FrameAnalysisResult(BaseModel):
    """Esquema estricto para la respuesta del análisis de cada fotograma."""
    
   
    visual_analysis: str = Field(
        description="Describe paso a paso lo que ves en la imagen basándote estrictamente en los píxeles, prestando especial atención a si el objeto solicitado está presente."
    )
    
    confidence: float = Field(
        description="Nivel de confianza de 0.0 a 1.0 basado en el análisis visual previo."
    )
    
    
    match_found: bool = Field(
        description="True si, basándote en visual_analysis, el objeto está claramente presente. False en caso contrario."
    )