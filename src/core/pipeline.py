import os
import json
import glob # Para buscar archivos 

from utils.video_utils import VideoLoader
from core.image_processor import VLMProcessor

class VLMPipeline:
    def __init__(self, base_folder = "data"):
        self.upload_dir = os.path.join(base_folder,"uploads")
        self.frames_dir = os.path.join(base_folder,"frames")
        self.results_dir = os.path.join(base_folder,"results")

        self.connector = VLMProcessor()

    def _limpiar_frames_antiguos(self):
        print(" Limpiando carpeta de frames antiguos...")
        
        archivos = glob.glob(os.path.join(self.frames_dir, "*.jpg"))
        
        for archivo in archivos:
            try:
                os.remove(archivo) 
            except Exception as e:
                print(f" No se pudo borrar {archivo}: {e}")        
    
    def process_video(self, file_name, prompt_usuario):

        self._limpiar_frames_antiguos()

        video_path = os.path.join(self.upload_dir,file_name)

        video_engine = VideoLoader(video_path, self.frames_dir)

        print("Extracción de frames ...")

        num_frames = video_engine.extract_frames(interval=0.5) #se obtiene los frames del video

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


