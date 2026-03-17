import asyncio
import os

from src.core.image_processor import VLMProcessor
from src.data.validators import FramesPath
from src.core.processing_strategies.base_strategy import ProcessingStrategy

from src.utils.file_utils import load_json
from src.utils.project_status import ProjectStatus

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


    async def procesar_cola(self, processor: VLMProcessor, prompt_usuario: str, cola: asyncio.Queue, resultados: list):
            
        FRAMES_PER_BATCH = self.config.get_video_int("frames_per_batch")
        flag = True

        while flag:
            frames_to_analyze = await self._extraer_lote_seguro(cola, FRAMES_PER_BATCH)

            if not frames_to_analyze: 
                print("Fin de la extracción detectado. Cerrando procesador.")
                break      

            if len(frames_to_analyze) < FRAMES_PER_BATCH :
                print("Iniciando proceso de ultimo batch (tamaño de este inferior al general)")
                flag = False

            ultimo_frame_id = frames_to_analyze[-1].frame_id 
            self.notify(ProjectStatus.ANALYZING, "Analizando...", ultimo_frame_id)

            await self._procesar_lote_interno(processor, prompt_usuario, frames_to_analyze, cola, resultados)

            # marcar las tareas como hechas 
            for _ in frames_to_analyze:
                cola.task_done()




    async def _extraer_lote_seguro(self, cola: asyncio.Queue, batch_size: int) -> list[FramesPath] :
        """
        Extrae un lote de frames de la cola 
        Retorna una lista de frames
        """
        lote = []
        count_frames = 0
        lista_vacia = False

        #mientras no se complete el lote ni se llegue al final
        while count_frames != batch_size and lista_vacia != True: 
        
            paquete = await cola.get() #extraer frame 
        
            if paquete is None: #se llego al final si se encuentra un none
                cola.task_done()
                cola.put_nowait(None) # necesario porq se saco de la cola y para detectarlo hay q devolverlo a ella 
                lista_vacia = True
                
            else:
                lote.append(paquete)
                count_frames=count_frames+1

        return lote




    async def _procesar_lote_interno(self, processor : VLMProcessor, prompt_usuario : str,  lote: list[FramesPath], cola : asyncio.Queue,
                                      resultados : list):
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