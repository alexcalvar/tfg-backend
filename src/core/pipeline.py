import os
import time
import asyncio
import datetime

from src.utils.file_utils import ensure_dir, save_json, load_json
from src.utils.project_status import ProjectStatus
from src.utils.video_utils import VideoLoader
from src.utils.config_loader import ConfigLoader

from src.core.image_processor import VLMProcessor

from src.data.validators import FramesPath

class VLMPipeline:
    def __init__(self, model_instance, provider_name, message_strategy):
        
        self.config = ConfigLoader()
        self.vlm = model_instance
        self.provider = provider_name
        self.message_strategy = message_strategy
        
        #  carga de prompts 
        self._load_prompts()

        #  estructura de carpetas 
        self._setup_project_directories()

        # inicialización del procesador
        self.processor = VLMProcessor(self.vlm, self.message_strategy, self.system_prompt, self.task_template)



    def _load_prompts(self):
        """aísla la lógica de lectura del archivo de prompts."""

        prompts_path = os.path.join(self.config.get_path("config_folder"), "prompts.json")
        config_prompts = load_json(prompts_path)

        lista_prompts = self.config.get_sys_config("selected_vlm_prompts_list")
        sys_prompt_key = self.config.get_sys_config("selected_sys_prompt")
        
        self.system_prompt = config_prompts[lista_prompts][sys_prompt_key]["system_instruction"]
        self.task_template = config_prompts[lista_prompts][sys_prompt_key]["task_template"]



    def _setup_project_directories(self):
        """crea el entorno de carpetas encapsulado para este proyecto."""
        
        projects_folder = self.config.get_path("projects_folder")
        run_id = f"project_{int(time.time())}"
        self.base_run_dir = os.path.join(projects_folder, run_id)

        ensure_dir(self.base_run_dir)
        
        self.video_dir = os.path.join(self.base_run_dir, "video")
        self.annotations_dir = os.path.join(self.base_run_dir, "annotations")
        self.frames_dir = os.path.join(self.base_run_dir, "frames")
        self.results_dir = os.path.join(self.base_run_dir, "results")
        
        ensure_dir(self.annotations_dir)
        ensure_dir(self.video_dir)
        ensure_dir(self.frames_dir)
        ensure_dir(self.results_dir)
        
        print(f" Nueva ejecución creada en: {self.base_run_dir}")



    async def process_video(self, source_video_path, prompt_usuario):
        """
        punto de entrada principal
        recibe la ruta absoluta del vídeo
        """
        file_name = os.path.basename(source_video_path)
        # el motor de vídeo ahora lee desde nuestra copia interna
        video_engine = VideoLoader(source_video_path, self.frames_dir)
        interval_time = self.config.get_video_float("frame_interval")

        # cacular total de frames a procesar
        total_frames = video_engine.get_expected_frame_count(interval_time)

        self._update_status(ProjectStatus.EXTRACTING, "Iniciando proceso de extracción de frames", 0, total_frames)

        cola_frames = asyncio.Queue()
        
        print("Extracción de frames ...")
        productor_task = asyncio.create_task(video_engine.extract_frames(interval_time, cola_frames))

        resultados_acumulados = []
        consumidor_task = asyncio.create_task(self._analizar_frames(cola_frames,total_frames, prompt_usuario, resultados_acumulados))

        await productor_task
        
        self._save_execution_config(file_name, prompt_usuario, total_frames)

        await cola_frames.join() 

        consumidor_task.cancel()

        self._update_status(ProjectStatus.COMPLETED, "Análisis finalizado con éxito.", total_frames, total_frames)

        try:
            #ordenar los frmaes del json
            resultados_acumulados.sort(key=lambda x: int(x["archivo"].split('_')[1].split('.')[0]))
        except Exception:
            # en caso de error devolver el archivo sin modificar
            pass 
       
        results_file_path = os.path.join(self.results_dir, "report.json")
        save_json(resultados_acumulados, results_file_path)
        print(f" Informe guardado en: {self.results_dir}")

        self._update_status(ProjectStatus.COMPLETED, "Análisis finalizado con éxito.", total_frames, total_frames)



    def _save_execution_config(self, video_filename, user_query, total_frames):
        """genera el archivo que almacena la información del proyecto con todos sus parámetros."""
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
                "frame_interval_seconds": self.config.get_video_float("frame_interval"), 
                "total_frames" : total_frames,
                "video_source": video_filename,
                "frames_batch" : self.config.get_video_int("frames_per_batch")
            }
        }
        
        config_path = os.path.join(self.base_run_dir, "execution_config.json")
        save_json(config_data, config_path)



    async def _analizar_frames(self, cola_frames: asyncio.Queue, total_frames: int, prompt_usuario: str, resultados: list):
        
        FRAMES_PER_BATCH = self.config.get_video_int("frames_per_batch")
        flag = True

        while flag:
           #  Obtener datos 
            frames_to_analyze = await self._extraer_lote_seguro(cola_frames, FRAMES_PER_BATCH)

            if not frames_to_analyze: #lista vacia
                print("Fin de la extracción detectado. Cerrando procesador.")
                break      

            #ultimo batch incompleto
            if len(frames_to_analyze) < FRAMES_PER_BATCH :
                print("Iniciando proceso de ultimo batch (tamaño de este inferior al general)")
                flag = False
                

            #  delegar el procesamiento pesado
            await self._procesar_lote(frames_to_analyze, total_frames, prompt_usuario, resultados, cola_frames)

            # liberar la memoria de la cola correspondiente a los frames procesados 
            for _ in frames_to_analyze:
                cola_frames.task_done()
            


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



    async def _procesar_lote(self, lote: list[FramesPath], total_frames: int, prompt_usuario: str, resultados: list, cola: asyncio.Queue):
            """Envía un lote al VLM y delega la validación de las respuestas."""
            try:
                ultimo_frame_id = lote[0].frame_id 
                self._update_status(ProjectStatus.ANALYZING, f"Analizando lote de {len(lote)} frames...", ultimo_frame_id, total_frames)

                # llamada a la IA
                respuestas_lote = await asyncio.to_thread(self.processor.analyze_frame, prompt_usuario, lote)

                # si el modelo siguió el estándar del objeto raíz, extraer la lista
                if "resultados" in respuestas_lote:
                    respuestas_lote = respuestas_lote["resultados"]

                # verificar q devuelva una lista con los resultados de cada frame y que el tamalo de la lista sea igual al numero de frames enviados 
                    if isinstance(respuestas_lote, list) and len(respuestas_lote) == len(lote):
                        for frame_obj, respuesta_dict in zip(lote, respuestas_lote):
                            await self._evaluar_frame_individual(frame_obj, respuesta_dict, resultados, cola)
                
                # Si el modelo alucinó y no devolvió una lista
                    else:
                        await self._gestionar_fallo_lote(lote, resultados, cola, "El modelo no devolvió una lista coherente.")

            except Exception as e: 
                # si se cae internet o la API da timeout, salvamos el lote
                print(f" Error de conexión analizando el lote: {e}")
                await self._gestionar_fallo_lote(lote, resultados, cola, f"Error de sistema/conexión: {e}")


    async def _evaluar_frame_individual(self, frame_obj: FramesPath, respuesta_dict: dict, resultados: list, cola: asyncio.Queue):
        """Valida el JSON de un único frame. Lo guarda si es correcto o gestiona su reintento si falla."""
        MAX_INTENTOS_COLA = self.config.get_video_int("max_intents_frame")

        if isinstance(respuesta_dict, dict) and "detectado" in respuesta_dict and "descripcion" in respuesta_dict:
            respuesta_dict["archivo"] = f"frame_{frame_obj.frame_id}.jpg"
            resultados.append(respuesta_dict)
            print(f" Terminado: frame_{frame_obj.frame_id}")
        
        else:
            actual_intent = frame_obj.intentos - 1

            if actual_intent <= 0:
                resultados.append({
                    "detectado": False,
                    "descripcion": f"Error: El modelo no generó un JSON válido tras {MAX_INTENTOS_COLA} intentos.",
                    "archivo": f"frame_{frame_obj.frame_id}.jpg"
                })
                print(f"  [ERROR] Frame {frame_obj.frame_id} falló demasiadas veces. Guardando defecto.")
            else:
                reintentar_paquete = FramesPath(frame_id=frame_obj.frame_id, frame_path=frame_obj.frame_path, intentos=actual_intent)
                print(f" [ADVERTENCIA] JSON inválido frame {frame_obj.frame_id}. Intentos restantes: {actual_intent}")
                await cola.put(reintentar_paquete)


    async def _gestionar_fallo_lote(self, lote: list[FramesPath], resultados: list, cola: asyncio.Queue, motivo: str):
        """Re-encola un lote completo si hubo un error crítico de formato o red, descontando un intento."""
        print(f" [ALERTA] Salvando lote debido a: {motivo}")
        
        for frame_obj in lote:
            actual_intent = frame_obj.intentos - 1
            
            if actual_intent > 0:
                await cola.put(FramesPath(frame_id=frame_obj.frame_id, frame_path=frame_obj.frame_path, intentos=actual_intent))
            else:
                resultados.append({
                    "detectado": False,
                    "descripcion": f"Error Crítico: {motivo}",
                    "archivo": f"frame_{frame_obj.frame_id}.jpg"
                })



    def _update_status(self, state: ProjectStatus, message: str, current_frame: int = 0, total_frames: int = 0):
        """escribe el estado actual del proceso en el disco en tiempo real."""
        status_data = {
            "state": state.value, 
            "message": message,
            "progress": {
                "current_frame": current_frame,
                "total_frames": total_frames
            },
            "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        status_path = os.path.join(self.base_run_dir, "status.json")
        save_json(status_data, status_path)