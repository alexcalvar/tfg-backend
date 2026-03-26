import os
import asyncio
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

from src.data.enums import StrategyType
from src.core.pipeline import VLMPipeline
from src.core.factories.model_factory import ModelFactory
from src.core.factories.algorithm_factory import AlgorithmFactory
from src.core.factories.processing_factory import ProcessingFactory
from src.data.dataset_loader import DatasetLoader

# Importamos tu clase que hace los cálculos matemáticos (Precision, Recall, etc.)
from src.evaluation.benchmark_runner import BenchmarkRunner

class AutomatedBenchmarkSuite:
    def __init__(self):
        load_dotenv()
        self.dataset_loader = DatasetLoader()
        
        # Configuración de rutas (Ajusta si la estructura de carpetas cambia)
        self.directorio_raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.gt_file = os.path.join(self.directorio_raiz, "datasets", "benchmarks", "ground_truth.json")

        self.output_dir = os.path.join(self.directorio_raiz, "datasets", "benchmarks", "reports")
        
        # Aseguramos que la carpeta de reportes exista
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.resultados_globales = []

    # ==========================================
    # 1. BATERÍA DE PRUEBAS 
    # ==========================================
    def obtener_experimentos(self) -> list[dict]:
        """Define aquí todas las combinaciones que quieres probar para tu TFG."""
        return [
            {
                "id_experimento": "EXP_01_QWEN_BATCH",
                "video": "video_perro_prueba.mp4",
                "prompt": "Dime si ves un perro en la imagen",
                "proveedor": "ollama",
                "modelo": "qwen_local",
                "estrategia": StrategyType.BATCH.value,
                "apply_alg" : False
            }
        ]

    # ==========================================
    # 2. MOTOR DE EJECUCIÓN AUTÓNOMO
    # ==========================================
    async def ejecutar_suite(self):
        experimentos = self.obtener_experimentos()
        total = len(experimentos)
        
        print("\n" + "="*60)
        print(f"  INICIANDO SUITE AUTOMATIZADA DE BENCHMARK ({total} pruebas)")
        print("="*60)

        for i, exp in enumerate(experimentos, start=1):
            print(f"\n [{i}/{total}] Ejecutando: {exp['id_experimento']}")
            print(f"   Modelo: {exp['modelo']} | Estrategia: {exp['estrategia']}")
            
            try:
                # 1. Fabricamos las dependencias específicas de este experimento
                algoritmo_normalizacion = AlgorithmFactory().create_algorithm(exp["apply_alg"])
                vlm_model, msg_strategy = ModelFactory().load_vlm(exp["proveedor"], exp["modelo"])
                process_strategy = ProcessingFactory().create_strategy(exp["estrategia"])
                
                # 2. Ensamblamos el Pipeline
                pipeline = VLMPipeline(
                    model_instance=vlm_model,
                    provider_name=exp["proveedor"],
                    message_strategy=msg_strategy,
                    processing_strategy=process_strategy,
                    temporal_normalizer=algoritmo_normalizacion
                )
                
                # 3. Conectamos el Evaluador y lanzamos
                # ---validacion de rita ---
                ruta_video_absoluta = os.path.join(self.directorio_raiz, "datasets", "videos_test", exp["video"])
                
                if not os.path.exists(ruta_video_absoluta):
                    raise FileNotFoundError(f"No se encuentra el archivo de vídeo en: {ruta_video_absoluta}")
                
                # 3. Conectamos el Evaluador y lanzamos
                runner = BenchmarkRunner(self.dataset_loader, pipeline)
                resultados = await runner.evaluate_video(
                    video_filename=ruta_video_absoluta,  
                    prompt=exp["prompt"],
                    gt_file_path=self.gt_file,
                    gt_format="simple_json",
                )

                # 4. Guardamos las métricas en memoria para el CSV
                metricas = resultados.get("binary_metrics", {})
                
                print(f"   ÉXITO -> F1-Score: {metricas.get('f1_score', 0):.4f} | Precisión: {metricas.get('precision', 0):.4f}")
                
                self.resultados_globales.append({
                    "ID_Experimento": exp["id_experimento"],
                    "Modelo": exp["modelo"],
                    "Proveedor": exp["proveedor"],
                    "Estrategia": exp["estrategia"],
                    "Normalizado": "SÍ" if exp["apply_alg"] else "NO", 
                    "Video": exp["video"],
                    "Precision": round(metricas.get("precision", 0), 4),
                    "Recall": round(metricas.get("recall", 0), 4),
                    "F1_Score": round(metricas.get("f1_score", 0), 4),
                    "Estado": "Completado"
                })

            except Exception as e:
                # Si un modelo falla (ej. se cuelga Ollama), capturamos el error y pasamos al siguiente
                print(f"   ERROR CRÍTICO en {exp['id_experimento']}: {e}")
                self.resultados_globales.append({
                    "ID_Experimento": exp["id_experimento"],
                    "Modelo": exp["modelo"],
                    "Proveedor": exp["proveedor"],
                    "Estrategia": exp["estrategia"],
                    "Video": exp["video"],
                    "Precision": 0, "Recall": 0, "F1_Score": 0,
                    "Estado": f"Error: {str(e)}"
                })

        # ==========================================
        # 3. EXPORTACIÓN DE RESULTADOS
        # ==========================================
        self._exportar_informe()

    def _exportar_informe(self):
        if not self.resultados_globales:
            print(" [ALERTA] No hay resultados que exportar.")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"benchmark_report_{timestamp}.csv"
        ruta_reporte = os.path.join(self.output_dir, nombre_archivo)
        
        # Usamos Pandas para generar el archivo tabular
        df = pd.DataFrame(self.resultados_globales)
        df.to_csv(ruta_reporte, index=False, sep=";") # sep=";" facilita abrirlo en Excel en español
        
        print("\n" + "="*60)
        print(f"  SUITE FINALIZADA.")
        print(f"  Reporte consolidado guardado en:\n    {ruta_reporte}")
        print("="*60)

if __name__ == "__main__":
    suite = AutomatedBenchmarkSuite()
    asyncio.run(suite.ejecutar_suite())