import asyncio
import os

from src.core.image_processor import VLMProcessor
from src.data.validators import FramesPath
from src.core.processing_strategies.base_strategy import ProcessingStrategy

from src.utils.file_utils import load_json
from src.utils.project_status import ProjectStatus

class TemporalStrategy(ProcessingStrategy):

    def __init__(self):
        super().__init__()

    def load_prompts(self ) -> tuple[str, str]:
        prompts_path = os.path.join(self.config.get_path("config_folder"), "prompts.json")
        config_prompts = load_json(prompts_path)
        categoria_prompts = self.config.get_sys_config("selected_vlm_prompts_list")
        sys_prompt_key = self.config.get_sys_config("temporal_sys_prompt")
        
        system_prompt = config_prompts[categoria_prompts][sys_prompt_key]["system_instruction"]
        task_template = config_prompts[categoria_prompts][sys_prompt_key]["task_template"]
        return system_prompt, task_template


    async def procesar_cola(self, processor: VLMProcessor, prompt_usuario: str, cola: asyncio.Queue, resultados: list):
        try:
            buffer_frames : list[FramesPath] = []
            
            print(" [DEBUG] Iniciando Fase 1 (Extracción inicial)...")
            primera_extraccion = await self._extraer_lote(cola, 2)
            if not primera_extraccion:
                print(" [DEBUG] Extracción inicial vacía. Finalizando.")
                return
                
            buffer_frames.extend(primera_extraccion)

            # --- FASE 1: INICIO ---
            frame_objetivo = buffer_frames[0]
            self.notify(ProjectStatus.ANALYZING, "Analizando inicio...", frame_objetivo.frame_id)
            print(f" [DEBUG] Llamando a procesar Fase 1 para frame_{frame_objetivo.frame_id}...")
            await self._procesar_lote_interno(processor, prompt_usuario, buffer_frames, frame_objetivo, resultados)
            cola.task_done()

            # --- FASE 2: MEDIO ---
            print(" [DEBUG] Entrando a la Fase 2 (Bucle)...")
            while True:
                frame_to_analyze = await self._extraer_lote(cola, 1)
                
                if not frame_to_analyze: 
                    print(" [DEBUG] Bucle roto correctamente.")
                    break      

                buffer_frames.extend(frame_to_analyze) 
                frame_objetivo = buffer_frames[1] 
                self.notify(ProjectStatus.ANALYZING, "Analizando ventana...", frame_objetivo.frame_id)
                print(f" [DEBUG] Llamando a procesar Fase 2 para frame_{frame_objetivo.frame_id}...")
                await self._procesar_lote_interno(processor, prompt_usuario, buffer_frames, frame_objetivo, resultados)

                cola.task_done()
                buffer_frames.pop(0)

            # --- FASE 3: FINAL ---
            print(" [DEBUG] Entrando a la Fase 3 (Evaluación Final)...")
            if len(buffer_frames) > 1:
                frame_objetivo = buffer_frames[-1] 
                self.notify(ProjectStatus.ANALYZING, "Analizando final...", frame_objetivo.frame_id)
                print(f" [DEBUG] Llamando a procesar Fase 3 para frame_{frame_objetivo.frame_id}...")
                await self._procesar_lote_interno(processor, prompt_usuario, buffer_frames, frame_objetivo, resultados)
                cola.task_done()
                
            cola.task_done() # Marcamos el None
            print(" [DEBUG] ¡Estrategia terminada y cola liberada con éxito!")

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


    async def _extraer_lote(self, cola: asyncio.Queue, batch_size: int) -> list[FramesPath]:
        lote = []
        while len(lote) < batch_size: 
            print(" [DEBUG] Esperando paquete de la cola...")
            paquete = await cola.get() 

            if paquete is None:
                print(" [DEBUG] ¡None recibido! El vídeo ha terminado.")
                return []

            print(f" [DEBUG] Extraído frame_{paquete.frame_id}")
            lote.append(paquete)

        return lote

    
    async def _procesar_lote_interno(self, processor: VLMProcessor, prompt_usuario: str, buffer_frames: list[FramesPath], frame_objetivo: FramesPath, resultados: list):
        """Coordina el envío de la ventana temporal al modelo gestionando los reintentos síncronos."""
        max_intentos = self.config.get_video_int("max_intents_frame")
        

        for intento in range(1, max_intentos + 1):
            try:
                respuesta_bruta = await asyncio.to_thread(processor.analyze_frame, prompt_usuario, buffer_frames)
                respuesta_validada = self._normalizar_respuesta(respuesta_bruta, frame_objetivo.frame_id)

                resultados.append(respuesta_validada)
                print(f" Terminado análisis temporal para frame_{frame_objetivo.frame_id}")
                return 

            except Exception as e:
                print(f" [ALERTA] Intento {intento}/{max_intentos} fallido temporalmente para frame_{frame_objetivo.frame_id}: {e}")
                if intento == max_intentos:
                    print(f" [ERROR CRÍTICO] Se agotaron los intentos para el frame_{frame_objetivo.frame_id}")
                    resultados.append(self._generar_error_fallback(frame_objetivo.frame_id, max_intentos, str(e)))


    def _normalizar_respuesta(self, respuesta: any, frame_id: int) -> dict:
        """Desempaqueta, valida el esquema JSON y estandariza los tipos de datos."""
        
        # Desempaquetar si viene dentro de un diccionario "resultados"
        if isinstance(respuesta, dict) and "resultados" in respuesta:
            respuesta = respuesta["resultados"]

        # Extraer el primer elemento si el modelo devolvió una lista
        respuesta_dict = respuesta[0] if (isinstance(respuesta, list) and len(respuesta) > 0) else respuesta

        # Cláusulas de guarda (Fail Fast)
        if not isinstance(respuesta_dict, dict):
            raise ValueError(f"El formato no es un diccionario válido. Tipo recibido: {type(respuesta_dict)}")
            
        if "detectado" not in respuesta_dict or "descripcion" not in respuesta_dict:
            raise ValueError(f"Faltan claves obligatorias en el JSON. Recibido: {respuesta_dict}")

        # Estandarización de datos
        if isinstance(respuesta_dict["detectado"], str):
            respuesta_dict["detectado"] = respuesta_dict["detectado"].strip().lower() == "true"
            
        respuesta_dict["archivo"] = f"frame_{frame_id}.jpg"
        
        return respuesta_dict


    def _generar_error_fallback(self, frame_id: int, max_intentos: int, error_msg: str) -> dict:
        """Genera un diccionario de error estandarizado cuando fallan todos los intentos."""
        return {
            "detectado": False,
            "descripcion": f"Error: Fallo de análisis temporal tras {max_intentos} intentos. Detalle: {error_msg}",
            "archivo": f"frame_{frame_id}.jpg"
        }