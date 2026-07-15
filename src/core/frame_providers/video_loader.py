import cv2 
import os
import math
import asyncio

from src.utils.config_loader import ConfigLoader
from src.utils.file_utils import ensure_dir
from src.data.validators import FramesPath
from src.core.frame_providers.frame_provider import BaseFrameProvider
from src.utils.logger import get_logger

logger = get_logger(__name__)

class VideoLoader(BaseFrameProvider):

    def __init__(self, video_path: str, output_folder: str, interval: float):
        self.config = ConfigLoader()
        self.video_path = video_path
        self.output_folder = output_folder
        self.interval = interval
        ensure_dir(output_folder)

    async def extract_frames(self, cola_frames: asyncio.Queue, cancel_event: asyncio.Event = None) -> None:
        cap = cv2.VideoCapture(self.video_path)

        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            
            if fps > 0:
                duration = total_frames / fps
            else:
                duration = 0
                
            logger.info(f"Info de Vídeo -> FPS: {fps:.2f} | Frames Totales: {total_frames} | Duración: {duration:.2f}s")
            
            step = math.ceil(fps * self.interval)
            n_frame = 0
            count_frame = 0
            max_intents = self.config.get_video_int("max_intents_frame")

            resize_width = self.config.get_video_int("resize_width")
            resize_height = self.config.get_video_int("resize_height")

            while cap.isOpened():
                # --- PUNTO DE CONTROL DE CANCELACIÓN ---
                if cancel_event and cancel_event.is_set():
                    logger.warning("Cancelación cooperativa activada en VideoLoader. Deteniendo extracción de frames...")
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                frame_redimensionado = cv2.resize(frame, (resize_width, resize_height))
                filename = f"frame_{count_frame}.jpg"
                save_path = os.path.join(self.output_folder, filename)
                
                cv2.imwrite(save_path, frame_redimensionado)

                timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                paquete_frame = FramesPath(count_frame, save_path, max_intents, timestamp_sec)

                await cola_frames.put(paquete_frame)
                logger.debug(f"Guardado frame nº {count_frame} (Posición real: {n_frame} - Optimizado)")

                n_frame += step
                count_frame += 1

                cap.set(cv2.CAP_PROP_POS_FRAMES, n_frame)  
                await asyncio.sleep(0)  

        cap.release()
        logger.info(f"Extracción finalizada o abortada. Total de frames enviados a la cola: {count_frame}")
        
    def get_expected_frame_count(self) -> int:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return 0
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_video_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release() 
        
        if fps > 0 and total_video_frames > 0:
            step = math.ceil(fps * self.interval)
            if step > 0:
                return math.ceil(total_video_frames / step) 
        return 0
    
    def get_source_name(self) -> str:    
        return os.path.basename(self.video_path)
    
    def cleanup(self) -> None:
        try:
            if os.path.exists(self.video_path):
                video_dir = os.path.dirname(self.video_path)
                os.remove(self.video_path)
                logger.info(f"Archivo de vídeo eliminado correctamente: {self.video_path}")
                
                if os.path.isdir(video_dir) and not os.listdir(video_dir):
                    os.rmdir(video_dir)
                    logger.debug(f"Directorio de vídeo temporal eliminado: {video_dir}")
            else:
                logger.warning(f"No se encontró el archivo para eliminar en: {self.video_path}")
        except Exception as e:
            logger.error(f"Error al realizar la limpieza de recursos en VideoLoader: {e}")