import os
import time
import asyncio
import datetime

from src.utils.file_utils import ensure_dir, save_json, load_json
from src.utils.project_status import ProjectStatus
from src.utils.video_utils import VideoLoader
from src.core.image_processor import VLMProcessor
from src.utils.config_loader import ConfigLoader

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
        
        # Actualizamos la firma para guardar el nombre del archivo
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
                "video_source": video_filename
            }
        }
        
        config_path = os.path.join(self.base_run_dir, "execution_config.json")
        save_json(config_data, config_path)



    async def _analizar_frames(self, cola_frames : asyncio.Queue, total_frames : int , prompt_usuario, resultados : list):

        MAX_INTENTOS_COLA = self.config.get_video_int("max_intents_frame")

        while True:
        
            paquete = await cola_frames.get()
            
            if paquete is None:
                cola_frames.task_done()
                break  

            # desempaquetar 
            frame_path, n_frame, max_intents = paquete

            try:
                respuesta_dict = await asyncio.to_thread(self.processor.analyze_frame( prompt_usuario, frame_path))

                self._update_status(ProjectStatus.ANALYZING, "Analizando los frames extraídos del video", n_frame, total_frames)

                if "detectado" in respuesta_dict and "descripcion" in respuesta_dict:
                    # json correcto, se guarda 
                    respuesta_dict["archivo"] = f"frame_{n_frame}.jpg"
                    resultados.append(respuesta_dict)
                    print(f" Terminado: frame_{n_frame}")
                
                else:
                    # enviar al final de la lista 
                    actual_intent=max_intents-1

                    if max_intents == 0 :
                        respuesta_dict = ({
                            "detectado": False,
                            "descripcion": f"Error: El modelo no pudo generar un JSON válido tras {MAX_INTENTOS_COLA} reintentos en cola.",
                        })

                        respuesta_dict["archivo"] = f"frame_{n_frame}.jpg"
                        resultados.append(respuesta_dict)
                    
                        print(f"  [ERROR] Frame {n_frame} falló demasiadas veces. Guardando resultado por defecto.")
                    
                    else:
                        reintentar_paquete = (frame_path, n_frame, actual_intent)
                        print(f"El modelo no fue capaz de analizar el frame {n_frame}, intentos restantes : {actual_intent}")
                        await cola_frames.put(reintentar_paquete)

            #al vaciarse la cola se lanza esta excepcin
            except asyncio.CancelledError:
                break

            except Exception as e: 
                print(f" Error analizando el frame numero {n_frame}: {e}")
                resultados.append({
                    "detectado": False,
                    "descripcion": f"Error de conexión o sistema: {str(e)}",
                    "archivo": f"frame_{n_frame}.jpg"
                })
            
            finally:
                #  avisar a la cola de que esta extracción concreta ya se procesó 
                #  porque el put() cuenta como un elemento nuevo
                cola_frames.task_done()
            


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