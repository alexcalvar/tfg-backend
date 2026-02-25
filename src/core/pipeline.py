import os
import json
import time
import asyncio

from utils.video_utils import VideoLoader
from core.image_processor import VLMProcessor

class VLMPipeline:
    def __init__(self, model_instance, message_strategy,system_prompt,task_template, base_folder = "data_ejs", result_folder = "projects"):
        
        self.vlm = model_instance
        self.message_strategy = message_strategy

        self.upload_dir = os.path.join(base_folder,"videos_test")

        # Generamos un ID único basado en la hora actual 
        run_id = f"run_{int(time.time())}"
        
        self.base_run_dir = os.path.join(result_folder, run_id)
        self.frames_dir = os.path.join(self.base_run_dir,"frames")
        self.results_dir = os.path.join(self.base_run_dir,"results")
        
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f" Nueva ejecución creada en: {self.base_run_dir}")

        self.processor = VLMProcessor(self.vlm, self.message_strategy, system_prompt, task_template)

    async def _analizar_frames(self, cola_frames : asyncio.Queue, prompt_usuario, resultados : list):

        while True:
        
            paquete = await cola_frames.get()
            
            if paquete is None:
                # Importante: Marcar el None como procesado también si usas join() en el futuro
                cola_frames.task_done()
                break  

            # 3. Desempaquetar de forma segura
            frame_path, n_frame = paquete

            try:
                respuesta_dict = await asyncio.to_thread(
                    self.processor.analyze_frame, prompt_usuario, frame_path
                )
                respuesta_dict["archivo"] = f"frame_{n_frame}.jpg"
                resultados.append(respuesta_dict)
            
            except Exception as e:
                print(f" Error analizando el frame numero {n_frame}: {e}")
            
            finally:
                print(f"  Terminado: frame{n_frame}")
                cola_frames.task_done()
            


    async def process_video(self, file_name, prompt_usuario):

        video_path = os.path.join(self.upload_dir,file_name)
        video_engine = VideoLoader(video_path, self.frames_dir)
        interval_time = 0.5

        cola_frames = asyncio.Queue()
        
        print("Extracción de frames ...")
        productor_task = asyncio.create_task(video_engine.extract_frames(interval_time, cola_frames))

        resultados_acumulados = []

        consumidor_task = asyncio.create_task(self._analizar_frames(cola_frames, prompt_usuario, resultados_acumulados))

        await productor_task

        await cola_frames.put(None)

        await consumidor_task
       
        results_file_name = "report.json"
        results_file_path = os.path.join(self.results_dir, results_file_name)

        with open (results_file_path,"w",encoding="utf-8") as f:
            json.dump(resultados_acumulados, f, indent=4, ensure_ascii=False)

        print(f" Informe guardado en: {self.results_dir}")


