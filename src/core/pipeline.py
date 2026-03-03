import os
import json
import time
import asyncio

from utils.file_utils import ensure_dir, save_json
from utils.video_utils import VideoLoader
from core.image_processor import VLMProcessor
from utils.config_loader import ConfigLoader

class VLMPipeline:
    def __init__(self, model_instance, message_strategy,system_prompt,task_template, base_folder = "data_ejs", result_folder = "projects"):
        
        self.config = ConfigLoader()

        self.vlm = model_instance
        self.message_strategy = message_strategy

        self.upload_dir = os.path.join(base_folder,"videos_test")

        # Generamos un ID único basado en la hora actual 
        run_id = f"project_{int(time.time())}"
        
        self.base_run_dir = os.path.join(result_folder, run_id)
        self.annotations_dir = os.path.join(self.base_run_dir,"annotations")
        self.frames_dir = os.path.join(self.base_run_dir,"frames")
        self.logs_dir = os.path.join(self.base_run_dir,"logs")
        self.results_dir = os.path.join(self.base_run_dir,"results")
        
        ensure_dir(self.annotations_dir)
        ensure_dir(self.frames_dir)
        ensure_dir(self.logs_dir)
        ensure_dir(self.results_dir)
        
        print(f" Nueva ejecución creada en: {self.base_run_dir}")

        #guardar configuracion de la prueba 
        ruta_save_config = os.path.join(self.base_run_dir, "execution_config.json")
        self.config.export_config(ruta_save_config)

        self.processor = VLMProcessor(self.vlm, self.message_strategy, system_prompt, task_template)

    async def _analizar_frames(self, cola_frames : asyncio.Queue, prompt_usuario, resultados : list):

        MAX_INTENTOS_COLA = self.config.get_video_int("max_intents_frame")

        while True:
        
            paquete = await cola_frames.get()
            
            if paquete is None:
                cola_frames.task_done()
                break  

            # Desempaquetar 
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
            


    async def process_video(self, file_name, prompt_usuario):

        video_path = os.path.join(self.upload_dir,file_name)
        video_engine = VideoLoader(video_path, self.frames_dir)
        interval_time = self.config.get_video_float("frame_interval")

        cola_frames = asyncio.Queue()
        
        print("Extracción de frames ...")
        productor_task = asyncio.create_task(video_engine.extract_frames(interval_time, cola_frames))

        resultados_acumulados = []

        consumidor_task = asyncio.create_task(self._analizar_frames(cola_frames, prompt_usuario, resultados_acumulados))

        await productor_task

        await cola_frames.join() #esperar a q la cola se vacie  

        consumidor_task.cancel()

        try:
            resultados_acumulados.sort(key=lambda x: int(x["archivo"].split('_')[1].split('.')[0]))
        except Exception:
            pass # si falla al reordenar devolver la lista tal cual 
       
        results_file_name = "report.json"
        results_file_path = os.path.join(self.results_dir, results_file_name)

        save_json(resultados_acumulados, results_file_path)

        print(f" Informe guardado en: {self.results_dir}")


