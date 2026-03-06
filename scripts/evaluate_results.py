import os
import sys

from src.data.dataset_loader import DatasetLoader
from src.evaluation.benchmark_runner import BenchmarkRunner



def main():
    print("=== EVALUACIÓN RÁPIDA DE PROYECTOS (MODO DESACOPLADO) ===")

    
    project_folder = input("\nIntroduce el nombre exacto de la carpeta del proyecto (ej. project_1715000000): ").strip()
    ground_truth = "ground_truth.json"
    if not project_folder:
        print(" [ERROR] Debes introducir un nombre de proyecto válido.")
        return

    #  Construcción de rutas absolutas seguras
    DIRECTORIO_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROJECT_DIR = os.path.join(DIRECTORIO_RAIZ, "projects", project_folder,"results")
    GROUND_TRUTH_FILE = os.path.join(DIRECTORIO_RAIZ, "projects", project_folder,"annotations", ground_truth)
    
    DATASET_FORMAT = "simple_json"

    # Verificación de seguridad antes de arrancar
    if not os.path.exists(PROJECT_DIR):
        print(f" [ERROR CRÍTICO] No existe el directorio: {PROJECT_DIR}")
        print(" Asegúrate de haber escrito bien el nombre y de que el proyecto se ejecutó previamente.")
        return

    # Inicialización ultra-rápida (Sin cargar modelos VLM en la memoria RAM)
    print("\n [1/2] Inicializando el motor de evaluación...")
    dataset_loader = DatasetLoader()
    
    # Fíjate en la magia: le pasamos None al pipeline. No necesitamos la IA.
    runner = BenchmarkRunner(dataset_loader=dataset_loader, pipeline_instance=None)

    #  Ejecución del cruce de datos
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