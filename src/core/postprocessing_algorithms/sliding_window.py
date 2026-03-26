
import copy
from typing import List

from src.data.validators import FrameResults
from src.core.postprocessing_algorithms.temporal_normalizer import TemporalNormalizer

class SlidingWindowNormalizer(TemporalNormalizer):
    
    def __init__(self, apply_alg:bool):
        super().__init__(apply_alg)
        self.window_size = 5


    #algoritmo de ejemplo para probar integracion 
    def _apply_sliding_window(self, results: List[FrameResults]) -> List[FrameResults]:
        """
        Aplica un filtro estadístico de ventana deslizante por mayoría (moda)
        para eliminar falsos positivos o negativos aislados.
        """
        if not results:
            return []

        # clonar la lista para evaluar sobre la foto "fija" del estado original,
        # evitando que los propios cambios afecten a las evaluaciones vecinas
        cleaned_results = copy.deepcopy(results)
        total_frames = len(results)
        
        half_window = self.window_size // 2

        for i in range(total_frames):
            # definir los bordes seguros de la ventana 
            start_idx = max(0, i - half_window)
            end_idx = min(total_frames, i + half_window + 1)
            
            # extraemos los valores de detección de los vecinos en la lista original
            window_values = [results[j].detectado for j in range(start_idx, end_idx)]
            
            # criterio de decision
            trues_count = sum(window_values)
            falses_count = len(window_values) - trues_count
            
            # asignar el valor suavizado al frame en la lista limpia
            cleaned_results[i].detectado = trues_count > falses_count

        return cleaned_results

