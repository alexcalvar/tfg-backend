import cv2 
import os
import math

class VideoLoader:
    def __init__(self, video_path, output_folder):
        self.video_path = video_path
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True) 


    def extract_frames(self, interval):
        cap = cv2.VideoCapture(self.video_path)

        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)

            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            
            
            if fps > 0:
                duration = total_frames / fps
            else:
                duration = 0
                
            print(f" DEBUG INFO:")
            print(f"   - FPS detectados: {fps}")
            print(f"   - Frames Totales: {total_frames}")
            print(f"   - Duración calculada: {duration:.2f} segundos")
            
            step =math.ceil(fps * interval)
            
            #captura de fotogramas
            n_frame = 0
            count_frame = 0

            while cap.isOpened():

                #ret es un boolean q verifica q se obtuvo un frame , es decir, no se llego al final todavia
                ret, frame = cap.read()

                if not ret:
                    break

                # OPTIMIZACIÓN 
                # Redimensionamos a 640x360 para no saturar la RAM de Ollama
                frame_redimensionado = cv2.resize(frame, (640, 360))

                filename = f"frame_{count_frame}.jpg"
                save_path = os.path.join(self.output_folder, filename)
                
                cv2.imwrite(save_path, frame_redimensionado)

                print(f"guardado frame nº {count_frame} (Posición real: {n_frame} - Optimizado)")

                n_frame += step
                count_frame += 1

                #se decide utilizar cap.set para avanzar directametne al frame a analizar en vez de 
                #avanzar recorriendo todos los frames y solo seleccionando los esperados
                cap.set(cv2.CAP_PROP_POS_FRAMES, n_frame)    
        
        cap.release()

        return count_frame




# --- BLOQUE DE PRUEBA ---
if __name__ == "__main__":
    
    video = "data/uploads/video_prueba_coches.mp4"
    carpeta_salida = "data/frames"

    if os.path.exists(video):
        loader = VideoLoader(video, carpeta_salida)
        # Extraer 1 por segundo
        loader.extract_frames(interval=1) 
    else:
        print(f" No encuentro el vídeo en: {video}")
        print("Crea la carpeta data/uploads y pon un vídeo ahí.")