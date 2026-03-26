import asyncio
import os

from src.core.image_processor import VLMProcessor
from src.data.validators import FrameResults, FramesPath
from src.core.processing_strategies.base_strategy import ProcessingStrategy
from src.core.output_parsers.base_parser import BaseFrameParser

from src.utils.file_utils import load_json
from src.utils.project_status import ProjectStatus

class BatchStrategy(ProcessingStrategy):


    def __init__(self, parser: BaseFrameParser): 
        super().__init__(parser)



    def load_prompts(self) -> tuple[str, str]:
        prompts_path = os.path.join(self.config.get_path("config_folder"), "prompts.json")
        config_prompts = load_json(prompts_path)
        categoria_prompts = self.config.get_sys_config("selected_vlm_prompts_list")
        sys_prompt_key = self.config.get_sys_config("batch_sys_prompt")
        
        base_system_prompt = config_prompts[categoria_prompts][sys_prompt_key]["system_instruction"]
        task_template = config_prompts[categoria_prompts][sys_prompt_key]["task_template"]

        instrucciones_formato = self.parser.get_format_instructions()
        system_prompt_final = f"{base_system_prompt}\n\n{instrucciones_formato}"

        return system_prompt_final, task_template




    async def process_queue(self, processor: VLMProcessor, user_prompt: str, queue: asyncio.Queue, resultados: list):
            
        FRAMES_PER_BATCH = self.config.get_video_int("frames_per_batch")
        flag = True

        while flag:
            frames_to_analyze = await self._extract_batch(queue, FRAMES_PER_BATCH)

            if not frames_to_analyze: 
                print("Fin de la extracción detectado. Cerrando procesador.")
                break      

            if len(frames_to_analyze) < FRAMES_PER_BATCH :
                print("Iniciando proceso de ultimo batch (tamaño inferior al general)")
                flag = False

            ultimo_frame_id = frames_to_analyze[-1].frame_id 
            self.notify(ProjectStatus.ANALYZING, "Analizando batch...", ultimo_frame_id)

            await self._process_batch_interno(processor, user_prompt, frames_to_analyze, queue, resultados)

            # marcar las tareas como hechas 
            for _ in frames_to_analyze:
                queue.task_done()




    async def _extract_batch(self, queue: asyncio.Queue, batch_size: int) -> list[FramesPath]:
        batch = []

        while len(batch) < batch_size:
            item = await queue.get()

            if item is None:
                queue.task_done()
                break

            batch.append(item)

        return batch




    async def _process_batch_interno(self, processor: VLMProcessor, user_prompt: str, batch: list[FramesPath], queue: asyncio.Queue, resultados: list):
        try:

            #construir el formato de peticion adecuado para el tipo de procesamiento especifico
            layout_mensaje = self._build_model_request(user_prompt, batch)
            
            # se envia al modelo el tipo de peticion especifica y se espera un string de respuesta
            respuesta_bruta = await asyncio.to_thread(processor.process_layout, layout_mensaje )

            #  el parser asume toda la responsabilidad de leer el string
            # y transformarlo en una lista de objetos frameresult
            resultados_parseados = self.parser.parse_batch(respuesta_bruta, batch)

            # guardar resultados
            for frame_result in resultados_parseados:
                resultados.append(frame_result)
                print(f" Terminado: frame_{frame_result.frame_id}")

        except Exception as e: 
            print(f" Error analizando el batch o de formato: {e}")
            await self._handle_batch_failure(batch, resultados, queue, f"Fallo de sistema/parseo: {e}")




    async def _handle_batch_failure(self, batch: list[FramesPath], resultados: list, queue: asyncio.Queue, motivo: str):
        print(f" [ALERTA] Reintentando batch debido a: {motivo}")
        for frame_obj in batch:
            intentos_restantes = frame_obj.intentos - 1
            if intentos_restantes > 0:
                await queue.put(FramesPath(frame_obj.frame_id, frame_obj.frame_path, intentos_restantes))
            else:
                resultados.append(self._crear_resultado(frame_obj.frame_id, False, f"Error Crítico: {motivo}"))    




    def _build_model_request(self, user_prompt : str, batch : list[FramesPath]) -> list:
        layout_mensaje = []
        
        for frame in batch:
            layout_mensaje.append({"type": "text", "content": user_prompt})
            layout_mensaje.append({"type": "image", "content": frame})

        return layout_mensaje




    @staticmethod
    def _crear_resultado(frame_id: int, detectado: bool, descripcion: str) -> FrameResults:
        """Función de apoyo para crear fallbacks de error."""
        return FrameResults(
            frame_id=frame_id, 
            detectado=detectado,
            descripcion=descripcion
        )