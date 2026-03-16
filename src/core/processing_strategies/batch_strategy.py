import asyncio
import os

from src.core.image_processor import VLMProcessor
from src.data.validators import FramesPath
from src.core.processing_strategies.base_strategy import ProcessingStrategy

from src.utils.file_utils import load_json

class BatchStrategy(ProcessingStrategy):

    def __init__(self):
        super().__init__()

    def load_prompts(self ) -> tuple[str, str]:
        prompts_path = os.path.join(self.config.get_path("config_folder"), "prompts.json")
        config_prompts = load_json(prompts_path)
        categoria_prompts = self.config.get_sys_config("selected_vlm_prompts_list")
        sys_prompt_key = self.config.get_sys_config("selected_sys_prompt")
        
        system_prompt = config_prompts[categoria_prompts][sys_prompt_key]["system_instruction"]
        task_template = config_prompts[categoria_prompts][sys_prompt_key]["task_template"]
        return system_prompt, task_template


    async def procesar_lote(self, processor: VLMProcessor, prompt_usuario: str, lote: list[FramesPath], cola: asyncio.Queue, 
                            resultados: list):
        try:

            respuestas_lote = await asyncio.to_thread(processor.analyze_frame, prompt_usuario, lote)

            if isinstance(respuestas_lote, dict) and "resultados" in respuestas_lote:
                respuestas_lote = respuestas_lote["resultados"]

            if isinstance(respuestas_lote, list) and len(respuestas_lote) == len(lote):
                for frame_obj, respuesta_dict in zip(lote, respuestas_lote):
                    await self._evaluar_frame_individual(frame_obj, respuesta_dict, resultados, cola)
            else:
                await self._gestionar_fallo_lote(lote, resultados, cola, "El modelo no devolvió una lista coherente.")

        except Exception as e: 
            print(f" Error analizando el lote: {e}")
            await self._gestionar_fallo_lote(lote, resultados, cola, f"Error de sistema/conexión: {e}")


    async def _evaluar_frame_individual(self, frame_obj: FramesPath, respuesta_dict: dict, resultados: list, cola: asyncio.Queue):
        MAX_INTENTOS = self.config.get_video_int("max_intents_frame")

        if isinstance(respuesta_dict, dict) and "detectado" in respuesta_dict and "descripcion" in respuesta_dict:
            
            if isinstance(respuesta_dict["detectado"], str):
                respuesta_dict["detectado"] = respuesta_dict["detectado"].strip().lower() == "true"
                
            respuesta_dict["archivo"] = f"frame_{frame_obj.frame_id}.jpg"
            resultados.append(respuesta_dict)
            print(f" Terminado: frame_{frame_obj.frame_id}")
        else:
            intentos_restantes = frame_obj.intentos - 1
            if intentos_restantes <= 0:
                resultados.append({
                    "detectado": False,
                    "descripcion": f"Error: JSON inválido tras {MAX_INTENTOS} intentos.",
                    "archivo": f"frame_{frame_obj.frame_id}.jpg"
                })
            else:
                await cola.put(FramesPath(frame_obj.frame_id, frame_obj.frame_path, intentos_restantes))

    async def _gestionar_fallo_lote(self, lote: list[FramesPath], resultados: list, cola: asyncio.Queue, motivo: str):
        print(f" [ALERTA] Salvando lote debido a: {motivo}")
        for frame_obj in lote:
            intentos_restantes = frame_obj.intentos - 1
            if intentos_restantes > 0:
                await cola.put(FramesPath(frame_obj.frame_id, frame_obj.frame_path, intentos_restantes))
            else:
                resultados.append({"detectado": False, "descripcion": f"Error Crítico: {motivo}", "archivo": f"frame_{frame_obj.frame_id}.jpg"})