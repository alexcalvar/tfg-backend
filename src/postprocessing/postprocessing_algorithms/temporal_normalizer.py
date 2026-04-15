import os
from abc import  abstractmethod

from typing import Any, List

from src.data.validators import FrameResults, EventInterval
from src.utils.config_loader import ConfigLoader
from src.postprocessing.postprocessing_mode import PostProcessingStrategy
from src.utils.file_utils import save_results

class TemporalNormalizer(PostProcessingStrategy):

    def __init__(self, apply_alg : bool):
        self.config = ConfigLoader()
        self.interval_time = self.config.get_video_float("frame_interval")
        self.apply = apply_alg 

    def execute(self, raw_results: List[FrameResults], results_dir: str) -> Any:
        events = self.process_and_group(raw_results=raw_results)

        events_file_path = os.path.join(results_dir, "intervalos.json")
        save_results(events, events_file_path)
        
        return events

    def process_and_group(self, raw_results: List[FrameResults]) -> List[EventInterval]:

        # Fase 1
        if self.apply:
            cleaned_results = self._apply_algoritm(raw_results)
            events = self._extract_intervals(cleaned_results)
            
        # Fase 2
        else:
            events = self._extract_intervals(raw_results)
        
        return events

    @abstractmethod
    def _apply_algoritm(self, results: List[FrameResults]) -> List[FrameResults]:
        """
         Corrige los falsos positivos/negativos analizando los vecinos.
        """

    def _extract_intervals(self, cleaned_results: List[FrameResults]) -> List[EventInterval]:
        """
        Convierte rachas continuas de detectado=True en objetos EventInterval.
        """
        
        resultado_eventos : List[EventInterval]= []
        in_event = False
        event_id=0

        for frame in cleaned_results:
                if frame.detectado:
                    ultimo_frame_true = frame.frame_id # se guarda el ultimo true conocido
                
                    if not in_event:            
                      inicio_evento = frame.frame_id
                      in_event = True

                elif not frame.detectado and in_event:
                # el evento se cierra exactamente en el último frame válido
                    fin_evento = ultimo_frame_true 
                    in_event = False
                    resultado_eventos.append(self._save_event(event_id=event_id, inicio_evento=inicio_evento, fin_evento=fin_evento))
                    event_id += 1

        # cierre de seguridad por si el evento termina justo con el frame final del vídeo
        if in_event:
            resultado_eventos.append(self._save_event(event_id=event_id, inicio_evento=inicio_evento, fin_evento=ultimo_frame_true))

        return resultado_eventos


    
    def _save_event(self,event_id: int, inicio_evento : int, fin_evento: int) -> EventInterval :

        start_time = inicio_evento*self.interval_time
        end_time = fin_evento*self.interval_time

        return EventInterval( event_id=event_id,start_frame=inicio_evento,end_frame=fin_evento,
                             start_timestamp=start_time,end_timestamp=end_time)


