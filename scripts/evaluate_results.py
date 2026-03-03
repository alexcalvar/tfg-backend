import os
import sys

# ---  Conectar la carpeta scripts con la carpeta src ---
DIRECTORIO_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARPETA_SRC = os.path.join(DIRECTORIO_RAIZ, "src")
sys.path.append(CARPETA_SRC)

from data.dataset_loader import DatasetLoader
from evaluation.benchmark_runner import BenchmarkRunner



def main():
    print("=== EVALUACIÓN RÁPIDA DE PROYECTOS (MODO DESACOPLADO) ===")

    # 1. Interfaz interactiva para el usuario
    project_folder = input("\nIntroduce el nombre exacto de la carpeta del proyecto (ej. project_1715000000): ").strip()
    
    if not project_folder:
        print(" [ERROR] Debes introducir un nombre de proyecto válido.")
        return

    # 2. Construcción de rutas absolutas seguras
    DIRECTORIO_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROJECT_DIR = os.path.join(DIRECTORIO_RAIZ, "projects", project_folder)
    GROUND_TRUTH_FILE = os.path.join(DIRECTORIO_RAIZ, "datasets", "benchmarks", "ground_truth.json")
    
    DATASET_FORMAT = "simple_json"

    # Verificación de seguridad antes de arrancar
    if not os.path.exists(PROJECT_DIR):
        print(f" [ERROR CRÍTICO] No existe el directorio: {PROJECT_DIR}")
        print(" Asegúrate de haber escrito bien el nombre y de que el proyecto se ejecutó previamente.")
        return

    # 3. Inicialización ultra-rápida (Sin cargar modelos VLM en la memoria RAM)
    print("\n [1/2] Inicializando el motor de evaluación...")
    dataset_loader = DatasetLoader()
    
    # Fíjate en la magia: le pasamos None al pipeline. No necesitamos la IA.
    runner = BenchmarkRunner(dataset_loader=dataset_loader, pipeline_instance=None)

    # 4. Ejecución del cruce de datos
    print(f" [2/2] Cruzando el report.json con la Verdad Absoluta...")
    try:
        resultados = runner.evaluate_existing_project(
            project_dir=PROJECT_DIR,
            gt_file_path=GROUND_TRUTH_FILE,
            gt_format=DATASET_FORMAT
        )

        print("\n=== RESULTADOS INSTANTÁNEOS ===")
        print(f" Precisión (Precision): {resultados['binary_metrics']['precision']}")
        print(f" Sensibilidad (Recall): {resultados['binary_metrics']['recall']}")
        print(f" F1-Score:              {resultados['binary_metrics']['f1_score']}")
        print("===============================\n")

    except Exception as e:
        print(f"\n[ERROR CRÍTICO DURANTE LA EVALUACIÓN]: {e}")

if __name__ == "__main__":
    main()