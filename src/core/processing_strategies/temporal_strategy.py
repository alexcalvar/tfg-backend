import asyncio
import os

from src.core.image_processor import VLMProcessor
from src.data.validators import FramesPath
from src.core.processing_strategies.base_strategy import ProcessingStrategy

from src.utils.file_utils import load_json

class TemporalStrategy(ProcessingStrategy):

    def __init__(self):
        super().__init__()

    def load_prompts(self ) -> tuple[str, str]:
        prompts_path = os.path.join(self.config.get_path("config_folder"), "prompts.json")
        config_prompts = load_json(prompts_path)
        categoria_prompts = self.config.get_sys_config("")
        sys_prompt_key = self.config.get_sys_config("")
        
        system_prompt = config_prompts[categoria_prompts][sys_prompt_key]["system_instruction"]
        task_template = config_prompts[categoria_prompts][sys_prompt_key]["task_template"]
        return system_prompt, task_template


    async def procesar_lote(self, processor: VLMProcessor, prompt_usuario: str, lote: list[FramesPath], cola: asyncio.Queue, 
                            resultados: list):
        pass


    async def _evaluar_frame_individual(self, frame_obj: FramesPath, respuesta_dict: dict, resultados: list, cola: asyncio.Queue):
        pass

    async def _gestionar_fallo_lote(self, lote: list[FramesPath], resultados: list, cola: asyncio.Queue, motivo: str):
        pass