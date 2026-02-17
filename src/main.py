import os

from core.pipeline import VLMPipeline 

def main():

    video_name = "video_prueba_coches.mp4" 
    prompt = "Busca uno o más vehiculos rojos o de algun color similar como naranja en la carretera"

   
    ruta_video = os.path.join("data", "uploads", video_name)
    if not os.path.exists(ruta_video):
        print(f" ERROR: No encuentro el vídeo en: {ruta_video}")
        print("Revisa que el nombre sea correcto y esté en la carpeta data/uploads.")
        return

    print(" Arrancando sistema ...")

    try:
        
        pipeline = VLMPipeline() 
        
        # Ejecutamos
        pipeline.process_video(video_name, prompt)
    
    except Exception as e:
        print(f" Ocurrió un error inesperado: {e}")

if __name__ == "__main__":
    main()