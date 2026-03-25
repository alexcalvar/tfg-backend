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
        

    def load_prompts(self ) -> tuple[str, str]:
        prompts_path = os.path.join(self.config.get_path("config_folder"), "prompts.json")
        config_prompts = load_json(prompts_path)
        categoria_prompts = self.config.get_sys_config("selected_vlm_prompts_list")
        sys_prompt_key = self.config.get_sys_config("batch_sys_prompt")
        
        base_system_prompt = config_prompts[categoria_prompts][sys_prompt_key]["system_instruction"]
        task_template = config_prompts[categoria_prompts][sys_prompt_key]["task_template"]

        instrucciones_formato = self.parser.get_format_instructions()
        system_prompt_final = f"{base_system_prompt}\n\n{instrucciones_formato}"

        return system_prompt_final, task_template


    async def procesar_cola(self, processor: VLMProcessor, prompt_usuario: str, cola: asyncio.Queue, resultados: list):
            
        FRAMES_PER_BATCH = self.config.get_video_int("frames_per_batch")
        flag = True

        while flag:
            frames_to_analyze = await self._extraer_lote_seguro(cola, FRAMES_PER_BATCH)

            if not frames_to_analyze: 
                print("Fin de la extracción detectado. Cerrando procesador.")
                break      

            if len(frames_to_analyze) < FRAMES_PER_BATCH :
                print("Iniciando proceso de ultimo batch (tamaño inferior al general)")
                flag = False

            ultimo_frame_id = frames_to_analyze[-1].frame_id 
            self.notify(ProjectStatus.ANALYZING, "Analizando lote...", ultimo_frame_id)

            await self._procesar_lote_interno(processor, prompt_usuario, frames_to_analyze, cola, resultados)

            # marcar las tareas como hechas 
            for _ in frames_to_analyze:
                cola.task_done()


    async def _extraer_lote_seguro(self, cola: asyncio.Queue, batch_size: int) -> list[FramesPath]:
        """Extrae un lote gestionando correctamente el None y los reintentos."""
        lote = []
        while len(lote) < batch_size: 
            paquete = await cola.get() 
            if paquete is None: 
                cola.task_done()
                if cola.empty():
                    cola.put_nowait(None) 
                    break
                else:
                    cola.put_nowait(None)
                    await asyncio.sleep(0.1) 
            else:
                lote.append(paquete)
        return lote


    async def _procesar_lote_interno(self, processor: VLMProcessor, prompt_usuario: str, lote: list[FramesPath], cola: asyncio.Queue, resultados: list):
        try:

            # 1. Data-Driven Design: Construimos el Layout para la petición
            layout_mensaje = self._build_model_request(prompt_usuario, lote)
            
            # 2. El procesador solo envía el layout y devuelve el string crudo
            respuesta_bruta = await asyncio.to_thread(processor.analyze_frame, layout_mensaje )

            # 3. El Parser asume toda la responsabilidad de leer el string
            # y transformarlo en una lista de objetos FrameResults
            resultados_parseados = self.parser.parse_batch(respuesta_bruta, lote)

            # 4. Guardar los resultados exitosos
            for frame_result in resultados_parseados:
                resultados.append(frame_result)
                print(f" Terminado: frame_{frame_result.frame_id}")

        except Exception as e: 
            print(f" Error analizando el lote o de formato: {e}")
            await self._gestionar_fallo_lote(lote, resultados, cola, f"Fallo de sistema/parseo: {e}")



    async def _gestionar_fallo_lote(self, lote: list[FramesPath], resultados: list, cola: asyncio.Queue, motivo: str):
        print(f" [ALERTA] Reintentando lote debido a: {motivo}")
        for frame_obj in lote:
            intentos_restantes = frame_obj.intentos - 1
            if intentos_restantes > 0:
                await cola.put(FramesPath(frame_obj.frame_id, frame_obj.frame_path, intentos_restantes))
            else:
                resultados.append(self._crear_resultado(frame_obj.frame_id, False, f"Error Crítico: {motivo}"))    


    def _build_model_request(self, prompt_usuario : str, lote : list[FramesPath]) -> list:
        layout_mensaje = []
        
        for frame in lote:
            layout_mensaje.append({"type": "text", "content": prompt_usuario})
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