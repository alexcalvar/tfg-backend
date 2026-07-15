import os
import time
import asyncio
import datetime

from src.data.validators import FrameResults
from src.core.processing_strategies.base_strategy import ProcessingStrategy
from src.core.message_strategies.message_builders import MessageStrategy
from src.utils.file_utils import ensure_dir, save_json, save_results, delete_directory
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

    def __init__(self, model_instance, provider_name, frame_provider: BaseFrameProvider, message_strategy: MessageStrategy, 
                 processing_strategy: ProcessingStrategy, postprocessing_strategy: PostProcessingStrategy, base_run_dir: str):
        
        self.config = ConfigLoader()
        self.vlm = model_instance
        self.provider = provider_name
        
        self.frame_provider = frame_provider
        self.message_strategy = message_strategy
        self.processing_strategy = processing_strategy 
        
        self.system_prompt, self.task_template = self.processing_strategy.load_prompts()

        self.status_manager = ProjectStatusManager(base_run_dir)
        self.processing_strategy.attach(self.status_manager)

        self.processor = VLMProcessor(self.vlm, self.message_strategy, self.system_prompt)
        self.postprocessing_strategy = postprocessing_strategy

        self.base_run_dir = base_run_dir
        self.results_dir = os.path.join(base_run_dir, "results")



    async def process_video(self, prompt_usuario: str, cancel_event: asyncio.Event = None) -> None:
        
        file_name = self.frame_provider.get_source_name()
        total_frames = self.frame_provider.get_expected_frame_count()
        self.status_manager.total_frames = total_frames

        self.status_manager.update_status(ProjectStatus.EXTRACTING, "Iniciando proceso de extracción de frames", 0)

        cola_frames = asyncio.Queue()
        resultados_acumulados: list[FrameResults] = []
        
        logger.info("Iniciando extracción y análisis concurrente...")
        productor_task = asyncio.create_task(self.frame_provider.extract_frames(cola_frames, cancel_event))
        consumidor_task = asyncio.create_task(self._analizar_frames(cola_frames, prompt_usuario, resultados_acumulados, cancel_event))

        try:
            # esperamos a que el extractor termine o se detenga por cancelación
            await productor_task

            # si el usuario canceló durante la extracción, abortamos inmediatamente
            if cancel_event and cancel_event.is_set():
                logger.warning("Pipeline abortado durante la extracción debido a una solicitud de cancelación.")
                self.status_manager.update_status(ProjectStatus.ERROR, "Análisis cancelado por el usuario.", 0)
                consumidor_task.cancel()
                # esperamos a que el consumidor termine de verdad de cancelarse antes de borrar,
                # para no arriesgarnos a que recree archivos (p.ej. status.json) tras el borrado
                await asyncio.gather(consumidor_task, return_exceptions=True)
                self.frame_provider.cleanup()
                delete_directory(self.base_run_dir)

                return

            # limpieza normal del vídeo original una vez extraído todo correctamente
            self.frame_provider.cleanup()

            # píldora envenenada controlada si la extracción terminó con éxito de forma natural
            cola_frames.put_nowait(None)
            
            self._save_execution_config(file_name, prompt_usuario, total_frames)

            # Esperamos a que el consumidor termine, o usamos la tarea en vez de
            # cola_frames.join(): si el consumidor corta el bucle por cancelación
            # sin drenar los elementos restantes (incluida la píldora envenenada),
            # cola_frames.join() se queda esperando para siempre.
            await consumidor_task

            # verificación final por si cancelaron mientras los últimos elementos de la cola se procesaban
            if cancel_event and cancel_event.is_set():
                logger.warning("Pipeline abortado durante el procesamiento final de los elementos de la cola.")
                self.status_manager.update_status(ProjectStatus.ERROR, "Análisis cancelado por el usuario.", 0)
                if not consumidor_task.done():
                    consumidor_task.cancel()
                await asyncio.gather(consumidor_task, return_exceptions=True)
                delete_directory(self.base_run_dir)
                return

        except asyncio.CancelledError:
            logger.error("El pipeline de procesamiento sufrió una cancelación asíncrona forzada externa.")
            self.status_manager.update_status(ProjectStatus.ERROR, "Error crítico: Cancelación forzada del sistema.", 0)
            raise
        finally:
            # pase lo que pase, ninguna corrutina se queda colgada (evit memory leaks)
            if not productor_task.done():
                productor_task.cancel()
            if not consumidor_task.done():
                consumidor_task.cancel()
            # esperamos a que la cancelación se complete de verdad antes de devolver el control:
            # si no, el caller puede seguir adelante  mientras estas
            # tareas todavía están escribiendo en disco.
            await asyncio.gather(productor_task, consumidor_task, return_exceptions=True)

        # si todo ha ido bien, guardamos reportes y ejecutamos la post-procesación
        resultados_acumulados.sort(key=lambda x: x.frame_id)
        
        results_file_path = os.path.join(self.results_dir, "report.json")
        save_results(resultados_acumulados, results_file_path)
        logger.info(f"Informe de frames guardado exitosamente en: {self.results_dir}")

        self.postprocessing_strategy.execute(resultados_acumulados, self.results_dir)
        self.status_manager.update_status(ProjectStatus.COMPLETED, "Análisis finalizado con éxito.", total_frames)

    def _save_execution_config(self, file_name, user_query, total_frames):
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
                "total_frames" : total_frames,
                "data_source": file_name,
                "frames_batch" : self.config.get_video_int("frames_per_batch")
            }
        }
        config_path = os.path.join(self.base_run_dir, "execution_config.json")
        save_json(config_data, config_path)
        logger.info(f"Configuración de ejecución guardada en: {config_path}")

    async def _analizar_frames(self, cola_frames: asyncio.Queue, prompt_usuario: str, resultados: list, cancel_event: asyncio.Event = None):
        logger.info("Delegando consumo de la cola a la estrategia de procesamiento visual...")
        await self.processing_strategy.process_queue(self.processor, prompt_usuario, cola_frames, resultados, cancel_event)