import os
import time
import asyncio
import datetime

from src.data.validators import FrameResults
from src.core.processing_strategies.base_strategy import ProcessingStrategy
from src.core.message_strategies.message_builders import MessageStrategy
from src.utils.file_utils import ensure_dir, save_json, save_results
from src.utils.project_status import ProjectStatus
from src.utils.video_utils import VideoLoader
from src.utils.config_loader import ConfigLoader

from src.core.image_processor import VLMProcessor
from src.core.postprocessing_algorithms.temporal_normalizer import TemporalNormalizer

from src.observer.status_manager import ProjectStatusManager

class VLMPipeline:

    def __init__(self, model_instance, provider_name, message_strategy : MessageStrategy, 
                 processing_strategy: ProcessingStrategy, temporal_normalizer : TemporalNormalizer):
        
        self.config = ConfigLoader()
        self.vlm = model_instance
        self.provider = provider_name
        self.message_strategy = message_strategy
        
        self.processing_strategy = processing_strategy 
        
        # hacemos q la estrategia que sus prompts especificos
        self.system_prompt, self.task_template = self.processing_strategy.load_prompts()

        self._setup_project_directories()

        self.status_manager = ProjectStatusManager(self.base_run_dir)
        # suscribir al espectador a los eventos 
        self.processing_strategy.attach(self.status_manager)

        self.processor = VLMProcessor(self.vlm, self.message_strategy, self.system_prompt)
        self.normalizer = temporal_normalizer




    async def process_video(self, source_video_path, prompt_usuario : str):
        """
        punto de entrada principal, recibe la ruta absoluta del vídeo
        """
        file_name = os.path.basename(source_video_path)
        # el motor de vídeo ahora lee desde nuestra copia interna
        video_engine = VideoLoader(source_video_path, self.frames_dir)
        interval_time = self.config.get_video_float("frame_interval")

        # cacular total de frames a procesar y pasarselos al observer para q lo sepa
        total_frames = video_engine.get_expected_frame_count(interval_time)
        self.status_manager.total_frames = total_frames

        self.status_manager.update_status(ProjectStatus.EXTRACTING, "Iniciando proceso de extracción de frames",0)

        cola_frames = asyncio.Queue()
        
        print("Extracción de frames ...")
        productor_task = asyncio.create_task(video_engine.extract_frames(interval_time, cola_frames))

        resultados_acumulados : list[FrameResults] = []
        consumidor_task = asyncio.create_task(self._analizar_frames(cola_frames, prompt_usuario, resultados_acumulados))

        await productor_task

        # Orquestador avisa manualmente a la cola de que no hay más frames
        cola_frames.put_nowait(None)
        
        self._save_execution_config(file_name, prompt_usuario, total_frames)

        await cola_frames.join() 

        consumidor_task.cancel()

        #ordenar los frmaes del json
        resultados_acumulados.sort(key=lambda x: x.frame_id)
        
        results_file_path = os.path.join(self.results_dir, "report.json")
        save_results(resultados_acumulados, results_file_path)
        print(f" Informe guardado en: {self.results_dir}")

        
        #usuario decide aplicar algoritmo de normalizacion
        #normalizar los resultados
        #eventos_definidos = self.normalizer.process_and_group(resultados_acumulados)

        #**************************BLOQUE PARA PRUEBAS DEL ALGORITMO*******************#
        #de esta forma se ve el antes y despues de aplicar el algoritmo pero la finalidad es utilizar el metodo process_and_group()
        #que 
        if self.normalizer.apply:
            post_normalizacion = self.normalizer._apply_sliding_window(resultados_acumulados)

            normalizacion_file_path = os.path.join(self.results_dir, "postnormalizacion.json")
            save_results(post_normalizacion, normalizacion_file_path)
            print(f" Archivo tras aplicar algoritmo guardados en: {self.results_dir}")
            
            eventos_definidos = self.normalizer._extract_intervals(post_normalizacion)
        else:
            #si no se decide aplicar algoritmo de normalizacion
            eventos_definidos = self.normalizer._extract_intervals(resultados_acumulados)

        
        #******************************************************************************#

        events_file_path = os.path.join(self.results_dir, "intervalos.json")
        save_results(eventos_definidos, events_file_path)
        print(f" Intervalos guardados en: {self.results_dir}")

        self.status_manager.update_status(ProjectStatus.COMPLETED, "Análisis finalizado con éxito.", total_frames)



    def _setup_project_directories(self):
        """crea el entorno de carpetas encapsulado para este proyecto."""
        
        projects_folder = self.config.get_path("projects_folder")
        run_id = f"project_{int(time.time())}"
        self.base_run_dir = os.path.join(projects_folder, run_id)

        ensure_dir(self.base_run_dir)
        
        self.video_dir = os.path.join(self.base_run_dir, "video")
        self.annotations_dir = os.path.join(self.base_run_dir, "annotations")
        self.frames_dir = os.path.join(self.base_run_dir, "frames")
        self.results_dir = os.path.join(self.base_run_dir, "results")
        
        ensure_dir(self.annotations_dir)
        ensure_dir(self.video_dir)
        ensure_dir(self.frames_dir)
        ensure_dir(self.results_dir)
        
        print(f" Nueva ejecución creada en: {self.base_run_dir}")





    #revisar donde debe estar este metodo
    def _save_execution_config(self, video_filename, user_query, total_frames):
        """genera el archivo que almacena la información del proyecto con todos sus parámetros"""

        config_data = {
            "execution_metadata": {
                "project_id": os.path.basename(self.base_run_dir), 
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "initialized"
            },
            "model_configuration": {
                "model_object": str(self.vlm), 
                "provider": self.provider, 
                "system_prompt": self.system_prompt,
                "task_template": self.task_template 
            },
            "inference_parameters": {
                "user_query": user_query,
                "frame_interval_seconds": self.config.get_video_float("frame_interval"), 
                "total_frames" : total_frames,
                "video_source": video_filename,
                "frames_batch" : self.config.get_video_int("frames_per_batch")
            }
        }
        
        config_path = os.path.join(self.base_run_dir, "execution_config.json")
        save_json(config_data, config_path)




    async def _analizar_frames(self, cola_frames: asyncio.Queue, prompt_usuario: str, resultados: list):

        print("Delegando consumo de la cola a la estrategia de procesamiento ...")
        await self.processing_strategy.process_queue(self.processor,prompt_usuario, cola_frames, resultados)

            
