
import copy
from typing import List

from src.data.validators import FrameResults
from src.core.postprocessing_algorithms.temporal_normalizer import TemporalNormalizer

class SlidingWindowNormalizer(TemporalNormalizer):
    
    def __init__(self):
        super().__init__()
        self.window_size = 5


    #algoritmo de ejemplo para probar integracion 
    def apply_sliding_window(self, results: List[FrameResults]) -> List[FrameResults]:
        """
        Aplica un filtro estadístico de ventana deslizante por mayoría (moda)
        para eliminar falsos positivos o negativos aislados.
        """
        if not results:
            return []

        # Clonamos la lista para evaluar sobre la foto "fija" del estado original,
        # evitando que nuestros propios cambios afecten a las evaluaciones vecinas.
        cleaned_results = copy.deepcopy(results)
        total_frames = len(results)
        
        half_window = self.window_size // 2

        for i in range(total_frames):
            # 1. Definimos los bordes seguros de la ventana (Edge Cases)
            start_idx = max(0, i - half_window)
            end_idx = min(total_frames, i + half_window + 1)
            
            # 2. Extraemos los valores de detección de los vecinos en la lista ORIGINAL
            window_values = [results[j].detectado for j in range(start_idx, end_idx)]
            
            # 3. Criterio de decisión: Votación por mayoría simple
            trues_count = sum(window_values)
            falses_count = len(window_values) - trues_count
            
            # 4. Asignamos el valor suavizado al frame en la lista LIMPIA
            cleaned_results[i].detectado = trues_count > falses_count

        return cleaned_results

