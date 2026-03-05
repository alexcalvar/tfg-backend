import os
import asyncio

from dotenv import load_dotenv

from core.pipeline import VLMPipeline 
from core.model_manager import ModelManager
from utils.file_utils import load_json

async def main():

    print("--- INICIANDO SISTEMA DE ANÁLISIS DE VÍDEO TFG ---")
    
    #  Cargar variables de entorno 
    load_dotenv()
    
    #  Cargar configuraciones en memoria 
    rutas_modelos = "configs/models_config.json"
    config_modelos = load_json(rutas_modelos)

    #paramentros q se reciben en la peticion
    video_name = "video_perro_prueba.mp4" 
    user_prompt = "Dime si ves un perro en la imagen"

    ruta_video = os.path.join("datasets", "videos_test", video_name)
    if not os.path.exists(ruta_video):
        print(f" ERROR: No encuentro el vídeo en: {ruta_video}")
        print("Revisa que el nombre sea correcto y esté en la carpeta datasets/video_test.")
        return

    print(" Arrancando sistema ...")
    print("Seleccione el VLM que desea utilizar:")
    
    # Obtenemos la lista de todas los modelos del json
    modelos_disponibles = list(config_modelos["vlms"].keys())
    
    for i, clave_modelo in enumerate(modelos_disponibles, start=1):
        desc = config_modelos["vlms"][clave_modelo].get("model_string", clave_modelo)
        print(f"{i} - {desc}")
        
    n = input("\nIntroduzca el numero del modelo que desea usar: ")
    
  
    try:
        indice = int(n) - 1
        if indice < 0 or indice >= len(modelos_disponibles):
            raise ValueError()
            
        clave_seleccionada = modelos_disponibles[indice]
        datos_modelo = config_modelos["vlms"][clave_seleccionada]
        
        vlm_model_name = datos_modelo["model_string"]
        vlm_provider = datos_modelo["provider"]
        print(f"\n[INFO] Configuración cargada: {datos_modelo['descripcion']}")
        
    except (ValueError, IndexError):
        print(" [ERROR] Selección no válida. Saliendo del sistema...")
        return

    try:
        vlm_model,provider, strategy = ModelManager(vlm_provider, vlm_model_name).load_vlm()

        pipeline = VLMPipeline(vlm_model,provider, strategy) 
        
        # Ejecutamos
        await pipeline.process_video(ruta_video, user_prompt)
    
    except Exception as e:
        print(f" Ocurrió un error inesperado: {e}")

if __name__ == "__main__":
    asyncio.run(main())