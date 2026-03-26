from src.data.enums import NormalizerAlgorithm
from src.utils.config_loader import ConfigLoader
from src.core.postprocessing_algorithms.temporal_normalizer import TemporalNormalizer
from src.core.postprocessing_algorithms.sliding_window import SlidingWindowNormalizer


class AlgorithmFactory:
    
    def __init__(self):
        self.config = ConfigLoader()

    
    def create_algorithm(self, apply : bool) -> TemporalNormalizer:
        """ Método encargado de crear la instancia del algoritmo de postprocesado"""

        algorithm_selected = self.config.get_sys_config("normalizer_algorithm")

        try:
            algorithm_selected = NormalizerAlgorithm(algorithm_selected)
        except ValueError:
            raise ValueError(f"Algoritmo de postprocesamiento no soportado: {algorithm_selected}")
        
        match algorithm_selected:
            
            case NormalizerAlgorithm.SLIDINGWINDOW:
                return SlidingWindowNormalizer(apply)
            
            case _:
                raise ValueError(f"[ERROR] Algoritmo de postprocesamiento {algorithm_selected} no soportado ")
            