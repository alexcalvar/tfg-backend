import os
import time

from fastapi import UploadFile

from src.core.factories.algorithm_factory import AlgorithmFactory
from src.core.pipeline import VLMPipeline
from src.core.factories.model_factory import ModelFactory
from src.core.factories.processing_factory import ProcessingFactory
from src.core.frame_providers.video_loader import VideoLoader

from src.postprocessing.resums_logic.semantic_processor import SemanticAnalyzer

from src.utils.file_utils import ensure_dir
from src.utils.logger import get_logger
from src.utils.config_loader import ConfigLoader

from src.data.enums import PostProcessingStr

logger = get_logger(__name__)


class AnalysisOrchestrator:
    def __init__(self):
        self.config = ConfigLoader()

    def _create_workspace(self) -> dict:
        """Crea las carpetas y devuelve un diccionario con las rutas clave."""
        projects_folder = self.config.get_path("projects_folder")
        run_id = f"project_{int(time.time())}"
        base_run_dir = os.path.join(projects_folder, run_id)
        
        # definir rutas
        paths = {
            "run_id": run_id,
            "base_dir": base_run_dir,
            "video_dir": os.path.join(base_run_dir, "video"),
            "frames_dir": os.path.join(base_run_dir, "frames"),
            "results_dir": os.path.join(base_run_dir, "results")
        }
        
        # crear carpetas físicas
        ensure_dir(paths["base_dir"])
        ensure_dir(paths["video_dir"])
        ensure_dir(paths["frames_dir"])
        ensure_dir(paths["results_dir"])
        
        return paths
    


    async def setup_video_pipeline(self, video_file: UploadFile, user_prompt: str, vlm_provider: str, vlm_model: str, processing_mode: str, 
                                   postprocessing_type: PostProcessingStr,interval_time : float,apply_alg: bool = None, llm_provider: str = None, llm_model: str = None):
        """Orquesta toda la preparación y devuelve el pipeline ensamblado."""
        
        # crear el entorno de trabajo
        paths = self._create_workspace()
        
        #guardar video
        video_path = os.path.join(paths["video_dir"], video_file.filename)
        await self._save_uploaded_file(video_file, video_path)
        
        #construir el frameprovider
        interval = interval_time
        frame_provider = VideoLoader(video_path, paths["frames_dir"], interval)
        
        # construir modelos y estrategias
        vlm_instance, msg_strategy = ModelFactory().load_vlm(vlm_provider, vlm_model)
        proc_strategy = ProcessingFactory().create_strategy(processing_mode)
        
        match postprocessing_type:
            
            case PostProcessingStr.ALGORITHM:
                # Comprobación de identidad estricta: solo fallamos si es 'None'
                if apply_alg is None:
                    raise ValueError("El modo de eventos requiere seleccionar si quiere aplicar el algoritmo de postprocesamiento")
                post_strategy = AlgorithmFactory().create_algorithm(apply_alg, interval_time)
        
            case PostProcessingStr.SEMANTIC:
                # Validamos que nos hayan pasado los parámetros del LLM
                if not llm_provider or not llm_model:
                    raise ValueError("El modo semántico requiere 'llm_provider' y 'llm_model'.")
                
                llm_instance = ModelFactory().load_llm(llm_provider, llm_model)
                post_strategy = SemanticAnalyzer(llm_instance, user_prompt, interval_time)
            
            case _:
                error_msg = f"Estrategia de postprocesamiento no implementada: {postprocessing_type}"
                logger.error(error_msg)
                raise NotImplementedError(error_msg)
            
        # ensamblar el pipeline inyectando
        pipeline = VLMPipeline(
            model_instance=vlm_instance,
            provider_name=vlm_provider,
            frame_provider=frame_provider,
            message_strategy=msg_strategy,
            processing_strategy=proc_strategy,
            postprocessing_strategy=post_strategy,
            base_run_dir=paths["base_dir"] # Se lo pasamos ya hecho
        )
        
        return pipeline, paths["run_id"]
    
    

    async def _save_uploaded_file(self, upload_file: UploadFile, destination_path: str):
        """Helper para guardar el archivo físico."""
        with open(destination_path, "wb") as buffer:
            content = await upload_file.read()
            buffer.write(content)