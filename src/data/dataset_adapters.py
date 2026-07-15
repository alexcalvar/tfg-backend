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
    Traduce un Ground Truth admitiendo dos formatos:

    1. Formato simple (diccionario clave -> booleano):
    {
      "frame_0.jpg": true,
      "frame_1.jpg": false
    }

    2. Formato "report.json" (la misma lista que devuelve el pipeline en
       results/report.json), para poder anotar la verdad copiando el
       report generado por la IA y corrigiendo a mano el campo 'detectado':
    [
      {"frame_id": 0, "detectado": true, "descripcion": "..."},
      {"frame_id": 1, "detectado": false, "descripcion": "..."}
    ]
    """
    def parse(self, file_path: str) -> dict[str, GroundTruthFrame]:
        self._check_file(file_path)

        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        diccionario_universal = {}

        if isinstance(raw_data, list):
            # Formato "report.json": lista de objetos con 'frame_id' y 'detectado'
            for item in raw_data:
                frame_id = item.get("frame_id")
                if frame_id is None:
                    continue

                frame_name = f"frame_{frame_id}.jpg"
                diccionario_universal[frame_name] = GroundTruthFrame(
                    frame_id=frame_name,
                    is_positive=bool(item.get("detectado", False))
                )
        else:
            # Formato simple: { "frame_X.jpg": true/false }
            for frame_name, is_present in raw_data.items():
                # Traducimos los datos asquerosos del archivo a nuestro Contrato Universal
                diccionario_universal[frame_name] = GroundTruthFrame(
                    frame_id=frame_name,
                    is_positive=bool(is_present)
                )

        return diccionario_universal

#  añadir un CSVAdapter, un YOLOAdapter, etc...