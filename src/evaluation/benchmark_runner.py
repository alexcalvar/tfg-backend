import os
from src.data.dataset_loader import DatasetLoader
from src.data.validators import FrameEvaluation
from src.evaluation.metrics_calculator import BinaryMetricsCalculator
from src.utils.file_utils import load_json, save_json
from src.evaluation.reporters import MetricsReporter

class BenchmarkRunner:
  
    def __init__(self, dataset_loader: DatasetLoader, pipeline_instance=None):
        self.dataset_loader = dataset_loader
        self.pipeline = pipeline_instance
        self.metrics_calculator = BinaryMetricsCalculator()

    async def evaluate_video(self, video_filename: str, prompt: str, gt_file_path: str, gt_format: str):
        """Enfoque 1: End-to-End. Procesa el vídeo desde cero y luego lo evalúa."""
        if not self.pipeline:
            raise ValueError("[ERROR CRÍTICO] Necesitas instanciar el pipeline VLM para procesar un vídeo.")
            
        print(f"\n--- INICIANDO BENCHMARK END-TO-END: {video_filename} ---")
        
        # FASE DE INFERENCIA
        video_path = os.path.join("datasets", "videos_test", video_filename)
        
        await self.pipeline.process_video(video_path, prompt)
        
        # FASE DE EVALUACIÓN (Delegada al método privado)
        ruta_report_ia = os.path.join(self.pipeline.results_dir, "report.json")
        return self._calcular_y_guardar(ruta_report_ia, gt_file_path, gt_format, self.pipeline.results_dir)

    def evaluate_existing_project(self, project_dir: str, gt_file_path: str, gt_format: str):
        """Enfoque 2: Desacoplado. Evalúa un report.json generado en el pasado."""
        print(f"\n--- EVALUANDO PROYECTO EXISTENTE: {project_dir} ---")
        ruta_report_ia = os.path.join(project_dir, "report.json")
        
        if not os.path.exists(ruta_report_ia):
            raise FileNotFoundError(f"[ERROR CRÍTICO] No se encontró el archivo {ruta_report_ia}")
            
        # FASE DE EVALUACIÓN (Delegada al mismo método privado)
        return self._calcular_y_guardar(ruta_report_ia, gt_file_path, gt_format, project_dir)

    # --- MÉTODO PRIVADO (El Motor Matemático) ---
    def _calcular_y_guardar(self, ruta_report_ia: str, gt_file_path: str, gt_format: str, output_dir: str):
        """Carga los JSONs, los fusiona y calcula las métricas. Reutilizable para ambos enfoques."""
        resultados_ia = load_json(ruta_report_ia) 
        ground_truth_dict = self.dataset_loader.load_ground_truth(gt_file_path, gt_format)
        
        evaluaciones = []
        for res_ia in resultados_ia:
            nombre_frame = res_ia["archivo"]
            
            if nombre_frame not in ground_truth_dict:
                print(f"  [WARNING] El {nombre_frame} no existe en el Ground Truth. Se ignora.")
                continue
                
            gt_frame = ground_truth_dict[nombre_frame]
            
            evaluacion = FrameEvaluation(
                frame_id=nombre_frame,
                ground_truth_detectado=gt_frame.is_positive,
                modelo_detectado=res_ia["detectado"]
            )
            evaluaciones.append(evaluacion)
        
        # Reseteamos la calculadora por si se usa varias veces en bucle
        self.metrics_calculator = BinaryMetricsCalculator()
        self.metrics_calculator.compute_matrix(evaluaciones)
        reporte_final = self.metrics_calculator.calculate_all_metrics()
        
        ruta_evaluacion_json = os.path.join(output_dir, "benchmark_metrics.json")
        save_json(reporte_final, ruta_evaluacion_json)
        
        # --- GENERACIÓN DEL PDF ---
        nombre_carpeta_proyecto = os.path.basename(output_dir)
        ruta_evaluacion_pdf = os.path.join(output_dir, "benchmark_report.pdf")
        MetricsReporter.generate_pdf(reporte_final, ruta_evaluacion_pdf, nombre_carpeta_proyecto)
        

        print(f" \n[ÉXITO] Benchmark completado.")
        print(f"  - JSON guardado en: {ruta_evaluacion_json}")
        print(f"  - PDF generado en:  {ruta_evaluacion_pdf}")

        return reporte_final