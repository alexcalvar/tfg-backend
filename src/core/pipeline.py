import os
import time
import asyncio
import datetime

from src.data.validators import FrameResults
from src.core.processing_strategies.base_strategy import ProcessingStrategy
from src.core.message_strategies.message_builders import MessageStrategy
from src.utils.file_utils import ensure_dir, save_json, save_results
from src.utils.project_status import ProjectStatus
from src.core.frame_providers.video_loader import VideoLoader
from src.utils.config_loader import ConfigLoader

from src.core.image_processor import VLMProcessor
from src.postprocessing.postprocessing_mode import PostProcessingStrategy
from src.core.frame_providers.frame_provider import BaseFrameProvider
from src.observer.status_manager import ProjectStatusManager

from src.utils.logger import get_logger

logger = get_logger(__name__)

class VLMPipeline:

    def __init__(self, model_instance, provider_name, frame_provider : BaseFrameProvider , message_strategy : MessageStrategy, 
                 processing_strategy: ProcessingStrategy, postprocessing_strategy : PostProcessingStrategy, base_run_dir: str):
        
        self.config = ConfigLoader()
        self.vlm = model_instance
        self.provider = provider_name
        
        self.frame_provider = frame_provider
        self.message_strategy = message_strategy
        self.processing_strategy = processing_strategy 
        
        # hacemos q la estrategia que sus prompts especificos
        self.system_prompt, self.task_template = self.processing_strategy.load_prompts()

        self.status_manager = ProjectStatusManager(base_run_dir)
        # suscribir al espectador a los eventos 
        self.processing_strategy.attach(self.status_manager)

        self.processor = VLMProcessor(self.vlm, self.message_strategy, self.system_prompt)
        self.postprocessing_strategy = postprocessing_strategy

        self.base_run_dir = base_run_dir
        self.results_dir = os.path.join(base_run_dir, "results")




    async def process_video(self, prompt_usuario : str):
        """
        punto de entrada principal, recibe la ruta absoluta del vídeo
        """
        file_name = self.frame_provider.get_source_name()
        # el motor de vídeo ahora lee desde nuestra copia interna
        #video_engine = VideoLoader(source_video_path, self.frames_dir)
        #interval_time = self.config.get_video_float("frame_interval")

        # cacular total de frames a procesar y pasarselos al observer para q lo sepa
        total_frames = self.frame_provider.get_expected_frame_count()
        self.status_manager.total_frames = total_frames

        self.status_manager.update_status(ProjectStatus.EXTRACTING, "Iniciando proceso de extracción de frames",0)

        cola_frames = asyncio.Queue()
        
        logger.info("Iniciando extracción de frames del vídeo...")
        productor_task = asyncio.create_task(self.frame_provider.extract_frames(cola_frames))

        resultados_acumulados : list[FrameResults] = []
        consumidor_task = asyncio.create_task(self._analizar_frames(cola_frames, prompt_usuario, resultados_acumulados))

        await productor_task

        #eliminar video una vez ya extraidos los frames
        self.frame_provider.cleanup()

        # orquestador avisa manualmente a la cola de que no hay más frames
        cola_frames.put_nowait(None)
        
        self._save_execution_config(file_name, prompt_usuario, total_frames)

        await cola_frames.join() 

        consumidor_task.cancel()

        #ordenar los frmaes del json
        resultados_acumulados.sort(key=lambda x: x.frame_id)
        
        results_file_path = os.path.join(self.results_dir, "report.json")
        save_results(resultados_acumulados, results_file_path)
        logger.info(f"Informe de frames guardado exitosamente en: {self.results_dir}")

        
        #usuario decide aplicar algoritmo de normalizacion
        #normalizar los resultados
        self.postprocessing_strategy.execute(resultados_acumulados, self.results_dir)

        self.status_manager.update_status(ProjectStatus.COMPLETED, "Análisis finalizado con éxito.", total_frames)



    #revisar donde debe estar este metodo
    def _save_execution_config(self, file_name, user_query, total_frames):
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
                "data_source": file_name,
                #HASH del video
                #Duracion
                #Git version
                #tipo
                #args_tipo
                "frames_batch" : self.config.get_video_int("frames_per_batch")
            }
        }
        
        config_path = os.path.join(self.base_run_dir, "execution_config.json")
        save_json(config_data, config_path)
        logger.info(f"Configuración de ejecución guardada en: {config_path}")




    async def _analizar_frames(self, cola_frames: asyncio.Queue, prompt_usuario: str, resultados: list):

        logger.info("Delegando consumo de la cola a la estrategia de procesamiento visual...")
        await self.processing_strategy.process_queue(self.processor,prompt_usuario, cola_frames, resultados)