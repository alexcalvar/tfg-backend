import os
import sys

# --- Conectar la carpeta scripts con la carpeta src ---
DIRECTORIO_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARPETA_SRC = os.path.join(DIRECTORIO_RAIZ, "src")
sys.path.append(CARPETA_SRC)

import asyncio
from dotenv import load_dotenv

from core.pipeline import VLMPipeline
from core.model_manager import ModelManager
from utils.file_utils import load_json

from data.dataset_loader import DatasetLoader
from evaluation.benchmark_runner import BenchmarkRunner

async def main():
    load_dotenv()
    print("=== INICIANDO ENTORNO DE EVALUACIÓN (BENCHMARK) ===")

    VIDEO_TEST = "video_perro_prueba.mp4" 
    
    DIRECTORIO_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    GROUND_TRUTH_FILE = os.path.join(DIRECTORIO_RAIZ, "datasets", "benchmarks", "ground_truth.json")
    RUTAS_MODELOS = os.path.join(DIRECTORIO_RAIZ, "configs", "models_config.json") # Ruta absoluta segura
    
    DATASET_FORMAT = "simple_json"
    SYS_PROMPT = "Analiza el frame. Responde con un JSON estricto que contenga 'detectado' (booleano) y 'descripcion' (texto)."
    USER_PROMPT = "ves un perro en la imagen"
    task_template = "Analiza la imagen y busca este elemento: '{user_query}'. Rellena los valores de este JSON:\n{\n  \"detectado\": true o false,\n  \"descripcion\": \"\", \n}\n\nInstrucciones de llenado:\nEn 'detectado' solo responderas true en caso de que hayas detectado el evento que se te pide buscar en la imagen \n- En 'descripcion' escribe una descripción del fotograma que se te pasa."

    # --- 1. SELECCIÓN DE MODELO ---
    config_modelos = load_json(RUTAS_MODELOS)
    modelos_disponibles = list(config_modelos["vlms"].keys())

    print("\n [1/3] Cargando catálogo de modelos de IA...")
    for i, clave_modelo in enumerate(modelos_disponibles, start=1):
        desc = config_modelos["vlms"][clave_modelo].get("model_string", clave_modelo)
        print(f"  {i} - {desc}")
        
    n = input("\nIntroduzca el numero del modelo que desea usar: ")
    
    try:
        indice = int(n) - 1
        if indice < 0 or indice >= len(modelos_disponibles):
            raise ValueError()
            
        clave_seleccionada = modelos_disponibles[indice]
        datos_modelo = config_modelos["vlms"][clave_seleccionada]
        
        vlm_model_name = datos_modelo["model_string"]
        vlm_provider = datos_modelo["provider"]
        print(f"\n[INFO] Configuración cargada: {datos_modelo.get('descripcion', vlm_model_name)}")
        
    except (ValueError, IndexError):
        print(" [ERROR] Selección no válida. Saliendo del sistema...")
        return

    # --- 2. INICIALIZACIÓN ---
    try:
        print(" [2/3] Inicializando Pipeline y Dataset Loader...")
        # Instanciamos todo pero NO ejecutamos el proceso de video manualmente
        vlm_model, strategy = ModelManager(vlm_provider, vlm_model_name).load_vlm()
        pipeline = VLMPipeline(vlm_model, strategy, SYS_PROMPT, task_template) 
        dataset_loader = DatasetLoader()
    except Exception as e:
        print(f" [ERROR CRÍTICO] Fallo al inicializar el modelo o componentes: {e}") 
        return # Cortamos la ejecución si algo falla

    # --- 3. EJECUCIÓN DEL BENCHMARK ---
    print(" [3/3] Conectando el Benchmark Runner y lanzando evaluación...")
    runner = BenchmarkRunner( dataset_loader, pipeline)

    try:
        # El Runner se encarga de llamar al pipeline internamente
        resultados = await runner.evaluate_video(
            video_filename=VIDEO_TEST,
            prompt=USER_PROMPT,
            gt_file_path=GROUND_TRUTH_FILE,
            gt_format=DATASET_FORMAT
        )
        
        print("\n=== RESUMEN DE RENDIMIENTO DE LA IA ===")
        print(f" Precisión (Precision): {resultados['binary_metrics']['precision']}")
        print(f" Sensibilidad (Recall): {resultados['binary_metrics']['recall']}")
        print(f" F1-Score:              {resultados['binary_metrics']['f1_score']}")
        print("=======================================")

    except Exception as e:
        print(f"\n[ERROR CRÍTICO DURANTE EL BENCHMARK]: {e}")

if __name__ == "__main__":
    asyncio.run(main())