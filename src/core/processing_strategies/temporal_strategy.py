import asyncio
import os

from src.core.image_processor import VLMProcessor
from src.data.validators import FramesPath, FrameResults
from src.core.processing_strategies.base_strategy import ProcessingStrategy

from src.core.output_parsers.base_parser import BaseFrameParser

from src.utils.file_utils import load_json
from src.utils.project_status import ProjectStatus


INITIAL_CONTEXT_FRAMES  = 2 # frame actual  y futuro para el primer frame
WINDOW_STEP  = 1            # avanzamos de 1 en 1 para mantener el solapamiento (t-1, t, t+1) centrandonos en analizar t

class TemporalStrategy(ProcessingStrategy):


    def __init__(self, parser : BaseFrameParser):
        super().__init__(parser)



    def load_prompts(self) -> tuple[str, str]:
        prompts_path = os.path.join(self.config.get_path("config_folder"), "prompts.json")
        config_prompts = load_json(prompts_path)
        categoria_prompts = self.config.get_sys_config("selected_vlm_prompts_list")
        sys_prompt_key = self.config.get_sys_config("temporal_sys_prompt")
        
        base_system_prompt = config_prompts[categoria_prompts][sys_prompt_key]["system_instruction"]
        task_template = config_prompts[categoria_prompts][sys_prompt_key]["task_template"]
        
        # añadri intrucciones especificas del parser correspondiente al prompt
        instrucciones_formato = self.parser.get_format_instructions()
        system_prompt_final = f"{base_system_prompt}\n\n{instrucciones_formato}"
        
        return system_prompt_final, task_template




    async def process_queue(self, processor: VLMProcessor, user_prompt: str, queue: asyncio.Queue, resultados: list) -> None:

        try:
            frame_buffer: list[FramesPath] = []
            # proceso de recorrer la queue de n frames
            # extraer los dos primeros frames de la queue para analizar el primero con el contexto de ambos
            if not await self._process_initial_phase(frame_buffer, processor, user_prompt, queue, resultados):
                return
            # recorre en grupos de tres para analizar el frame t ayudandose de t-1 y t+1 
            await self._process_middle_phase(frame_buffer, processor, user_prompt, queue, resultados)

            # ultima iteracion del proceso donde se analiza el ultmio frame ayudandose del frame n-1 
            await self._process_final_phase(frame_buffer, processor, user_prompt, queue, resultados)

        except Exception as e:
            
            print(f"\n  [ERROR FATAL OCULTO EN LA ESTRATEGIA] Motivo: {e}")
            
            # liberar queue para que no se quede congelado
            print(" [DEBUG] Forzando liberación de la queue por emergencia...")
            while not queue.empty():
                try:
                    queue.get_nowait()
                    queue.task_done()

                except Exception as ex:
                    print(f"Error liberando la queue {ex}")
            queue.task_done() 




    async def _process_initial_phase(self, frame_buffer : list[FramesPath], processor: VLMProcessor, user_prompt: str, queue: asyncio.Queue, resultados: list) -> bool:
        print("[AVISO] Comenzando fase de procesamiento : TEMPORAL-STRATEGY")
        primera_extraccion = await self._extract_batch(queue, INITIAL_CONTEXT_FRAMES)
        if not primera_extraccion:
             return False
                
        frame_buffer.extend(primera_extraccion)
        await self._evaluar_ventana(processor, user_prompt, frame_buffer, 0, "Analizando inicio", queue, resultados)
        
        return True




    async def _process_middle_phase(self,frame_buffer : list[FramesPath], processor: VLMProcessor, user_prompt: str, queue: asyncio.Queue, resultados: list):
        while True:
            frame_to_analyze = await self._extract_batch(queue, WINDOW_STEP)
            if not frame_to_analyze: 
                break      

            frame_buffer.extend(frame_to_analyze) 
                
            # el frame objetivo es el del medio 
            await self._evaluar_ventana(processor, user_prompt, frame_buffer, 1, "Analizando ventana...", queue, resultados)
            
            frame_buffer.pop(0) # eliminar el frame mas a la izq de la ventana 




    async def _process_final_phase(self,frame_buffer : list[FramesPath], processor: VLMProcessor, user_prompt: str, queue: asyncio.Queue, resultados: list):
        if len(frame_buffer) > 1:
            # el frame objetivo es el último , por eso indice -1
            await self._evaluar_ventana(processor, user_prompt, frame_buffer, -1, "Analizando final...", queue, resultados)
                
            queue.task_done() # marcar el none final del bucle

    


    async def _evaluar_ventana(self, processor: VLMProcessor, user_prompt: str, 
                               frame_buffer: list[FramesPath], target_index: int, 
                               status_message: str, queue: asyncio.Queue, resultados: list) -> None:
        """Método para evaluar una ventana de frames y actualizar/notificar el progreso"""
        
        target_frame = frame_buffer[target_index] 

        self.notify(ProjectStatus.ANALYZING, status_message, target_frame.frame_id)
        
        print(f" [DEBUG] {status_message} - frame_{target_frame.frame_id}")
        
        await self._process_batch_interno(processor, user_prompt, frame_buffer, target_frame, resultados)

        queue.task_done()




    async def _extract_batch(self, queue: asyncio.Queue, batch_size: int) -> list[FramesPath]:
        batch = []
        while len(batch) < batch_size: 
        
            frame_item = await queue.get() 

            if frame_item is None:
                print(" [DEBUG] Flag de fin recibida. El vídeo ha terminado.")
                # se vuelve a meter el none para no romper el join() del orquestador
                queue.task_done()

                queue.put_nowait(None) 
                
                break

            batch.append(frame_item)

        return batch




    async def _process_batch_interno(self, processor: VLMProcessor, user_prompt: str, frame_buffer: list[FramesPath],
                                                                     target_frame: FramesPath, resultados: list) -> None:
        """Coordina el envío de la ventana de frames al modelo gestionando los reintentos de forma secuencial"""

        max_attempts = self.config.get_video_int("max_intents_frame")
        
        for attempt  in range(1, max_attempts + 1):
            try:
                #construir el formato de peticion adecuado para el tipo de procesamiento especifico
                layout_mensaje = self._build_model_request(user_prompt, frame_buffer)
                
                # enviar al procesador el mensaje 
                respuesta_bruta = await asyncio.to_thread(processor.process_layout, layout_mensaje)
                
                # utilizar el parser especifico para validar la respuesta en el formato esperado
                respuesta_validada = self.parser.parse(respuesta_bruta, target_frame.frame_id)

                resultados.append(respuesta_validada)
                print(f" Terminado análisis temporal para frame_{target_frame.frame_id}")
                return 

            except Exception as e:
                print(f" [AVISO] Intento {attempt}/{max_attempts} fallido procesando el frame_{target_frame.frame_id}: {e}")
                if attempt == max_attempts:
                    print(f" [ERROR] Se agotaron los intentos para el frame_{target_frame.frame_id}")
                    resultados.append(self._generar_error_fallback(target_frame.frame_id, max_attempts, str(e)))




    def _build_model_request(self, user_prompt: str, frame_buffer: list[FramesPath]) -> list[dict]:
        """Construye el formato de peticion especifico del tipo de procesamiento para la petición que se enviará al modelo"""
        layout_mensaje = [{"type": "text", "content": user_prompt}]
        
        for frame in frame_buffer:
            layout_mensaje.append({"type": "image", "content": frame})
            
        return layout_mensaje




    @staticmethod
    def _generar_error_fallback(frame_id: int, max_intentos: int, error_msg: str) -> FrameResults:
        """Estandariza el formato de respuesta fallida que se almacena en el archivo de resultados"""
        return FrameResults(frame_id=frame_id, 
                            detectado=False, 
                            descripcion=f"Error: Fallo de análisis temporal tras {max_intentos} intentos. Detalle: {error_msg}")