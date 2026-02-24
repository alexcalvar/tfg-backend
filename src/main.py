import os
import json
from dotenv import load_dotenv

from core.pipeline import VLMPipeline 
from core.adapters.message_builders import CloudMessageBuilder,LocalMessageBuilder
from core.model_manager import ModelManager

def load_json_file ( ruta_archivo):
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(f"no se encuentra el archivo {ruta_archivo}")
    with open(ruta_archivo, "r", encoding="utf-8") as file:
        return json.load(file)

def main():

    print("--- INICIANDO SISTEMA DE ANÁLISIS DE VÍDEO TFG ---")
    
    # 1. Cargar variables de entorno 
    load_dotenv()
    
    # 2. Cargar configuraciones en memoria 
    rutas_modelos = "configs/models_config.json"
    rutas_prompts = "configs/prompts.json"
    
    config_modelos = load_json_file(rutas_modelos)
    config_prompts = load_json_file(rutas_prompts)

    video_name = "video_perro_prueba.mp4" 
    prompt = "Dime si ves un perro en la imagen"

    task_template = config_prompts["vlm"]["frame_analysis_2"]["task_template"]
    sys_prompt = config_prompts["vlm"]["frame_analysis_2"]["system_instruction"]

   
    ruta_video = os.path.join("data", "videos_test", video_name)
    if not os.path.exists(ruta_video):
        print(f" ERROR: No encuentro el vídeo en: {ruta_video}")
        print("Revisa que el nombre sea correcto y esté en la carpeta data/uploads.")
        return

    print(" Arrancando sistema ...")
    print("Seleccione el VLM que desea utilizar:")
    
    # 1. Obtenemos la lista de todas las claves (modelos) del JSON
    modelos_disponibles = list(config_modelos["vlms"].keys())
    
    # 2. Generamos el menú dinámicamente con un bucle
    for i, clave_modelo in enumerate(modelos_disponibles, start=1):
        desc = config_modelos["vlms"][clave_modelo].get("descripcion", clave_modelo)
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
        vlm_model, strategy = ModelManager(vlm_provider, vlm_model_name).load_vlm()

        pipeline = VLMPipeline( vlm_model,strategy, sys_prompt, task_template) 
        
        # Ejecutamos
        pipeline.process_video(video_name, prompt)
    
    except Exception as e:
        print(f" Ocurrió un error inesperado: {e}")

if __name__ == "__main__":
    main()