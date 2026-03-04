import os
import time
import asyncio
import datetime
import shutil # Para copiar el vídeo

from utils.file_utils import ensure_dir, save_json, load_json
from utils.video_utils import VideoLoader
from core.image_processor import VLMProcessor
from utils.config_loader import ConfigLoader

class VLMPipeline:
    def __init__(self, model_instance, provider_name, message_strategy):
        
        self.config = ConfigLoader()
        self.vlm = model_instance
        self.provider = provider_name
        self.message_strategy = message_strategy
        
        #  carga de prompts encapsulada
        self._load_prompts()

        #  estructura de carpetas aislada
        self._setup_project_directories()

        # inicialización del procesador
        self.processor = VLMProcessor(self.vlm, self.message_strategy, self.system_prompt, self.task_template)


    def _load_prompts(self):
        """Aísla la lógica de lectura del archivo de prompts."""
        prompts_path = os.path.join(self.config.get_path("config_folder"), "prompts.json")
        config_prompts = load_json(prompts_path)

        lista_prompts = self.config.get_sys_config("selected_vlm_prompts_list")
        sys_prompt_key = self.config.get_sys_config("selected_sys_prompt")
        
        self.system_prompt = config_prompts[lista_prompts][sys_prompt_key]["system_instruction"]
        self.task_template = config_prompts[lista_prompts][sys_prompt_key]["task_template"]


    def _setup_project_directories(self):
        """Crea el entorno de carpetas encapsulado para este proyecto."""
        
        result_folder = self.config.get_path("result_folder")
        run_id = f"project_{int(time.time())}"
        self.base_run_dir = os.path.join(result_folder, run_id)
        
        # 
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
        Punto de entrada principal
        Recibe la ruta absoluta del vídeo
        """
        # copiamos el vídeo original a la carpeta interna del proyecto
        file_name, internal_video_path = self._save_video(source_video_path)
        
        # el motor de vídeo ahora lee desde nuestra copia interna
        video_engine = VideoLoader(internal_video_path, self.frames_dir)
        interval_time = self.config.get_video_float("frame_interval")

        cola_frames = asyncio.Queue()
        
        print("Extracción de frames ...")
        productor_task = asyncio.create_task(video_engine.extract_frames(interval_time, cola_frames))

        resultados_acumulados = []
        consumidor_task = asyncio.create_task(self._analizar_frames(cola_frames, prompt_usuario, resultados_acumulados))

        await productor_task
        
        # Actualizamos la firma para guardar el nombre del archivo
        self._save_execution_config(file_name, prompt_usuario)

        await cola_frames.join() 
        consumidor_task.cancel()

        try:
            resultados_acumulados.sort(key=lambda x: int(x["archivo"].split('_')[1].split('.')[0]))
        except Exception:
            pass 
       
        results_file_path = os.path.join(self.results_dir, "report.json")
        save_json(resultados_acumulados, results_file_path)

        print(f" Informe guardado en: {self.results_dir}")


    def _save_video(self, source_video_path):
        """ Almacena el video en la carpeta correspondiente si no está ya allí."""
        file_name = os.path.basename(source_video_path)
        internal_video_path = os.path.join(self.video_dir, file_name)
        
        # copiamos si la ruta de origen no es exactamente la misma que el destino
        if os.path.abspath(source_video_path) != os.path.abspath(internal_video_path):
            shutil.copy2(source_video_path, internal_video_path)

        return file_name, internal_video_path

    def _save_execution_config(self, video_filename, user_query):
        """Genera el archivo que almacena la información del proyecto con todos sus parámetros."""
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
                "video_source": video_filename
            }
        }
        
        config_path = os.path.join(self.base_run_dir, "execution_config.json")
        save_json(config_data, config_path)


    async def _analizar_frames(self, cola_frames : asyncio.Queue, prompt_usuario, resultados : list):

        MAX_INTENTOS_COLA = self.config.get_video_int("max_intents_frame")

        while True:
        
            paquete = await cola_frames.get()
            
            if paquete is None:
                cola_frames.task_done()
                break  

            # desempaquetar 
            frame_path, n_frame, max_intents = paquete

            try:
                respuesta_dict = await asyncio.to_thread(
                    self.processor.analyze_frame, prompt_usuario, frame_path
                )

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
            
