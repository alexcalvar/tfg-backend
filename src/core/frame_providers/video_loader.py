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


    async def extract_frames(self, cola_frames : asyncio.Queue ) -> None:
        cap = cv2.VideoCapture(self.video_path)

        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            
            if fps > 0:
                duration = total_frames / fps
            else:
                duration = 0
                
            # Agrupamos la info en una sola línea de log elegante
            logger.info(f"Info de Vídeo -> FPS: {fps:.2f} | Frames Totales: {total_frames} | Duración: {duration:.2f}s")
            
            step = math.ceil(fps * self.interval)
            
            #captura de fotogramas
            n_frame = 0
            count_frame = 0
            max_intents = self.config.get_video_int("max_intents_frame") # numero maximo de intentos q se realizan en caso de error en un frame

            resize_width = self.config.get_video_int("resize_width")
            resize_height = self.config.get_video_int("resize_height")

            while cap.isOpened():

                #ret es un boolean q verifica q se obtuvo un frame , es decir, no se llego al final todavia
                ret, frame = cap.read()

                if not ret:
                    break

                # optimizar los frames
                # redimensionar a 640x360 para no saturar la ram con ollama
                frame_redimensionado = cv2.resize(frame, (resize_width,resize_height ))

                filename = f"frame_{count_frame}.jpg"
                save_path = os.path.join(self.output_folder, filename)
                
                cv2.imwrite(save_path, frame_redimensionado)

                # obtener el tiempo exacto en el que ocurre este frame en el vídeo
                timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                
                paquete_frame = FramesPath(count_frame,save_path, max_intents, timestamp_sec)

                await cola_frames.put(paquete_frame) # se almacena la ruta en la cola para q luego el procesador acceda a la ruta del framse

                # Usamos DEBUG para no inundar la consola si el vídeo es muy largo
                logger.debug(f"Guardado frame nº {count_frame} (Posición real: {n_frame} - Optimizado)")

                n_frame += step
                count_frame += 1

                #se decide utilizar cap.set para avanzar directametne al frame a analizar en vez de 
                #avanzar recorriendo todos los frames y solo seleccionando los esperados
                cap.set(cv2.CAP_PROP_POS_FRAMES, n_frame)  

                # obligamos al bucle a ceder el control 
                # esto permite que el consumidor envíe la petición a la api del modelo
                # sin esperar a que acabe el vídeo entero.
                await asyncio.sleep(0)  

        cap.release()
        logger.info(f"Extracción finalizada. Total de frames enviados a la cola: {count_frame}")
        

    def get_expected_frame_count(self) -> int:
        """calcula cuántos frames se extraerán antes de iniciar el proceso"""
        
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
        """
        Libera los recursos eliminando el archivo de vídeo original del disco 
        para evitar redundancia de datos y saturación del almacenamiento.
        """
        try:
            if os.path.exists(self.video_path):
                # Obtenemos el directorio antes de borrar el archivo
                video_dir = os.path.dirname(self.video_path)
                
                os.remove(self.video_path)
                logger.info(f"Archivo de vídeo eliminado correctamente: {self.video_path}")
                
                # Opcional: Intentamos eliminar la carpeta 'video' si ha quedado vacía
                if os.path.isdir(video_dir) and not os.listdir(video_dir):
                    os.rmdir(video_dir)
                    logger.debug(f"Directorio de vídeo temporal eliminado: {video_dir}")
            else:
                logger.warning(f"No se encontró el archivo para eliminar en: {self.video_path}")
        except Exception as e:
            logger.error(f"Error al realizar la limpieza de recursos en VideoLoader: {e}")