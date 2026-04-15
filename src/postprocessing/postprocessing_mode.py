from abc import ABC, abstractmethod
from typing import List, Any
from src.data.validators import FrameResults

class PostProcessingStrategy(ABC):
    """
    Contrato base (Interfaz) para cualquier funcionalidad que procese 
    los resultados crudos generados por el VLM.
    """
    
    @abstractmethod
    def execute(self, raw_results: List[FrameResults], results_dir: str) -> Any:
        """
        Ejecuta la lógica de post-procesamiento.
        
        Args:
            raw_results: Lista de objetos tipados con las detecciones del VLM.
            
        Returns:
            Any: Un objeto  tipado 
        """
        pass