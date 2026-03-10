from abc import ABC, abstractmethod
import json
import os
from src.data.validators import GroundTruthFrame

# --- LA INTERFAZ BASE ---
class BaseDatasetAdapter(ABC):
    """Plantilla estricta que todo traductor de dataset debe cumplir."""
    
    @abstractmethod
    def parse(self, file_path: str) -> dict[str, GroundTruthFrame]:
        pass

    def _check_file(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[ERROR] No se encuentra el dataset en: {file_path}")

# --- Formato JSON Simple ---
class SimpleJSONAdapter(BaseDatasetAdapter):
    """
    Traduce un JSON básico con este formato:
    {
      "frame_0.jpg": true,
      "frame_1.jpg": false
    }
    """
    def parse(self, file_path: str) -> dict[str, GroundTruthFrame]:
        self._check_file(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        diccionario_universal = {}
        
        for frame_name, is_present in raw_data.items():
            # Traducimos los datos asquerosos del archivo a nuestro Contrato Universal
            diccionario_universal[frame_name] = GroundTruthFrame(
                frame_id=frame_name,
                is_positive=bool(is_present)
            )
            
        return diccionario_universal

#  añadir un CSVAdapter, un YOLOAdapter, etc...