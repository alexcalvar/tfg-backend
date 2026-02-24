import os
import json
import time

from utils.video_utils import VideoLoader
from core.image_processor import VLMProcessor

class VLMPipeline:
    def __init__(self, model_instance, message_strategy,system_prompt,task_template, base_folder = "data", result_folder = "projects"):
        
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

        self.connector = VLMProcessor(self.vlm, self.message_strategy, system_prompt, task_template)


    def process_video(self, file_name, prompt_usuario):

        interval_time = 0.5

        video_path = os.path.join(self.upload_dir,file_name)

        video_engine = VideoLoader(video_path, self.frames_dir)

        print("Extracción de frames ...")

        num_frames = video_engine.extract_frames(interval_time) #se obtiene los frames del video

        print(f"generados {num_frames} frames del video : {file_name}")

        resultados_acumulados = []
        
        for i in range(0, num_frames):
            
            frame_path = f"{self.frames_dir}/frame_{i}.jpg"

            try:

                respuesta_dict = self.connector.analyze_frame(prompt_usuario, frame_path)

                respuesta_dict["archivo"] = f"frame_{i}.jpg"

                resultados_acumulados.append(respuesta_dict)
                print(f"  Terminado: frame{i}")

            except Exception as e:
                print(f" Error analizando el frame numero {i}: {e}")
        results_file_name = "report.json"
        results_file_path = os.path.join(self.results_dir, results_file_name)

        with open (results_file_path,"w",encoding="utf-8") as f:
            json.dump(resultados_acumulados, f, indent=4, ensure_ascii=False)

        print(f" Informe guardado en: {self.results_dir}")


