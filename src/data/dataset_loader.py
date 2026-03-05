from src.data.dataset_adapters import BaseDatasetAdapter, SimpleJSONAdapter
from src.data.validators import GroundTruthFrame

class DatasetLoader:
    def __init__(self):
        # El "Registro" de traductores disponibles
        self._adapters: dict[str, BaseDatasetAdapter] = {
            "simple_json": SimpleJSONAdapter(),
            # "yolo_txt": YOLOAdapter(),  
        }

    def load_ground_truth(self, file_path: str, format_type: str) -> dict[str, GroundTruthFrame]:
        """
        Carga un dataset externo y lo devuelve en el formato Universal del TFG.
        """
        if format_type not in self._adapters:
            formatos_validos = list(self._adapters.keys())
            raise ValueError(f"[ERROR CRÍTICO] Formato '{format_type}' no soportado. Usa uno de: {formatos_validos}")
        
        # 1. Selecciona al traductor adecuado
        traductor = self._adapters[format_type]
        print(f" [SISTEMA] Leyendo Ground Truth con el adaptador: {format_type}")
        
        # 2. Le pide que haga el trabajo sucio y devuelve los datos limpios
        return traductor.parse(file_path)