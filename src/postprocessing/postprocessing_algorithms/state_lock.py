
import copy
from typing import List

from src.data.validators import FrameResults
from src.postprocessing.postprocessing_algorithms.temporal_normalizer import TemporalNormalizer

class StateLockNormalizer(TemporalNormalizer):

    def __init__(self, apply_alg: bool, interval: float):
        super().__init__(apply_alg, interval)

    def _apply_algoritm(self, results: List[FrameResults]) -> List[FrameResults]:
        
        total_frames = len(results)
        if total_frames < 2:
            return copy.deepcopy(results)

        cleaned_results = copy.deepcopy(results)

        # buscar el primer par de frames consecutivos con el mismo valor,
        # que fija el estado inicial bloqueado
        start_idx = None
        state = None
        for i in range(total_frames - 1):
            if results[i].detectado == results[i + 1].detectado:
                start_idx = i
                state = results[i].detectado
                break

        # nunca hubo dos frames consecutivos iguales: se mantiene todo tal cual
        if start_idx is None:
            return cleaned_results

        idx = start_idx + 2
        while idx < total_frames:
            if results[idx].detectado == state:
                idx += 1
                continue

            # frame con valor opuesto al estado bloqueado: comprobar si viene
            # acompañado de otro consecutivo igual
            # o si es un frame aislado 
            if idx + 1 < total_frames and results[idx + 1].detectado != state:
                state = results[idx].detectado
            else:
                cleaned_results[idx].detectado = state

            idx += 1

        return cleaned_results
