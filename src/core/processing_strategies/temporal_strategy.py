import asyncio
import os

from src.core.image_processor import VLMProcessor
from src.data.validators import FramesPath, FrameResults
from src.core.processing_strategies.base_strategy import ProcessingStrategy

from src.core.output_parsers.base_parser import BaseFrameParser

from src.utils.file_utils import load_json
from src.utils.project_status import ProjectStatus

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


    async def procesar_cola(self, processor: VLMProcessor, prompt_usuario: str, cola: asyncio.Queue, resultados: list):

        CONTEXTO_INICIAL_FRAMES = 2 # frame actual  y futuro para el primer frame
        PASO_VENTANA = 1            # avanzamos de 1 en 1 para mantener el solapamiento (t-1, t, t+1) centrandonos en analizar t

        try:
            buffer_frames: list[FramesPath] = []
            
            # --- FASE 1: INICIO ---
            primera_extraccion = await self._extraer_lote(cola, CONTEXTO_INICIAL_FRAMES)
            if not primera_extraccion:
                return
                
            buffer_frames.extend(primera_extraccion)
            await self._evaluar_ventana(processor, prompt_usuario, buffer_frames, 0, "Analizando inicio...", cola, resultados)

            # --- FASE 2: MEDIO ---
            while True:
                frame_to_analyze = await self._extraer_lote(cola, PASO_VENTANA)
                if not frame_to_analyze: 
                    break      

                buffer_frames.extend(frame_to_analyze) 
                
                # El frame objetivo es el del medio (índice 1)
                await self._evaluar_ventana(processor, prompt_usuario, buffer_frames, 1, "Analizando ventana...", cola, resultados)
                
                buffer_frames.pop(0) # Avanzamos la ventana

            # --- FASE 3: FINAL ---
            if len(buffer_frames) > 1:
                # El frame objetivo es el último (índice -1)
                await self._evaluar_ventana(processor, prompt_usuario, buffer_frames, -1, "Analizando final...", cola, resultados)
                
            cola.task_done() # Marcamos el None final del bucle

        except Exception as e:
            # ¡EL CAZADOR DE ERRORES SILENCIOSOS!
            import traceback
            print(f"\n  [ERROR FATAL OCULTO EN LA ESTRATEGIA] ")
            print(f"Motivo: {e}")
            traceback.print_exc()
            
            # Forzamos la liberación de la cola para que el Orquestador no se quede congelado
            print(" [DEBUG] Forzando liberación de la cola por emergencia...")
            while not cola.empty():
                try:
                    cola.get_nowait()
                    cola.task_done()
                except:
                    pass
            cola.task_done() # Para asegurar

    async def _evaluar_ventana(self, processor: VLMProcessor, prompt_usuario: str, 
                               buffer_frames: list[FramesPath], target_index: int, 
                               mensaje_estado: str, cola: asyncio.Queue, resultados: list):
        """Método auxiliar genérico para evaluar una ventana de frames y notificar el progreso."""
        
        frame_objetivo = buffer_frames[target_index] 
        self.notify(ProjectStatus.ANALYZING, mensaje_estado, frame_objetivo.frame_id)
        
        print(f" [DEBUG] {mensaje_estado} (frame_{frame_objetivo.frame_id})")
        
        await self._procesar_lote_interno(processor, prompt_usuario, buffer_frames, frame_objetivo, resultados)
        cola.task_done()

    async def _extraer_lote(self, cola: asyncio.Queue, batch_size: int) -> list[FramesPath]:
        lote = []
        while len(lote) < batch_size: 
            # print(" [DEBUG] Esperando paquete de la cola...")
            paquete = await cola.get() 

            if paquete is None:
                print(" [DEBUG] ¡None recibido! El vídeo ha terminado.")
                # Volvemos a meter el None para no romper el join() del orquestador
                # y para que otros bucles sepan que se acabó
                cola.task_done()

                cola.put_nowait(None) 
                
                # Salimos del bucle devolviendo lo que hayamos logrado recolectar
                break

            # print(f" [DEBUG] Extraído frame_{paquete.frame_id}")
            lote.append(paquete)

        return lote

    
    async def _procesar_lote_interno(self, processor: VLMProcessor, prompt_usuario: str, buffer_frames: list[FramesPath], frame_objetivo: FramesPath, resultados: list):
        """Coordina el envío de la ventana temporal al modelo gestionando los reintentos síncronos."""
        max_intentos = self.config.get_video_int("max_intents_frame")
        
        for intento in range(1, max_intentos + 1):
            try:
                # 1. Construimos el layout con la ventana cronológica
                layout_mensaje = self._build_model_request(prompt_usuario, buffer_frames)
                
                # 2. El procesador solo recibe el layout
                respuesta_bruta = await asyncio.to_thread(processor.analyze_frame, layout_mensaje)
                
                # 3. ¡El Parser inyectado hace la magia!
                respuesta_validada = self.parser.parse(respuesta_bruta, frame_objetivo.frame_id)

                resultados.append(respuesta_validada)
                print(f" Terminado análisis temporal para frame_{frame_objetivo.frame_id}")
                return 

            except Exception as e:
                print(f" [ALERTA] Intento {intento}/{max_intentos} fallido temporalmente para frame_{frame_objetivo.frame_id}: {e}")
                if intento == max_intentos:
                    print(f" [ERROR CRÍTICO] Se agotaron los intentos para el frame_{frame_objetivo.frame_id}")
                    resultados.append(self._generar_error_fallback(frame_objetivo.frame_id, max_intentos, str(e)))


    def _build_model_request(self, prompt_usuario: str, buffer_frames: list[FramesPath]) -> list[dict]:
        """Construye el layout Data-Driven para la petición temporal."""
        layout_mensaje = [{"type": "text", "content": prompt_usuario}]
        
        for frame in buffer_frames:
            layout_mensaje.append({"type": "image", "content": frame})
            
        return layout_mensaje

   
    @staticmethod
    def _generar_error_fallback(frame_id: int, max_intentos: int, error_msg: str) -> FrameResults:
        """Genera un diccionario de error estandarizado cuando fallan todos los intentos."""
        return FrameResults(frame_id=frame_id, 
                            detectado=False, 
                            descripcion=f"Error: Fallo de análisis temporal tras {max_intentos} intentos. Detalle: {error_msg}")