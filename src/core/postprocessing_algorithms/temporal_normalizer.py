import os
from abc import ABC, abstractmethod

from typing import List

from src.data.validators import FrameResults, EventInterval
from src.utils.config_loader import ConfigLoader


class TemporalNormalizer(ABC):

    def __init__(self):
        self.config = ConfigLoader()
        self.inteval_time = self.config.get_video_float("frame_interval")


    def process_and_group(self, raw_results: List[FrameResults]) -> List[EventInterval]:

        pass

        # Fase 1
        #cleaned_results = self._apply_sliding_window(raw_results)
        
        # Fase 2
        #events = self._extract_intervals(cleaned_results)
        
        #return events

    @abstractmethod
    def apply_sliding_window(self, results: List[FrameResults]) -> List[FrameResults]:
        """
         Corrige los falsos positivos/negativos analizando los vecinos.
        """

    def extract_intervals(self, cleaned_results: List[FrameResults]) -> List[EventInterval]:
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

        start_time = inicio_evento*self.inteval_time
        end_time = fin_evento*self.inteval_time

        return EventInterval( event_id=event_id,start_frame=inicio_evento,end_frame=fin_evento,
                             start_timestamp=start_time,end_timestamp=end_time)


