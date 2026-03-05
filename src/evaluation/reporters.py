from fpdf import FPDF
import os

class MetricsReporter:
    @staticmethod
    def generate_pdf(metrics_data: dict, output_path: str, project_name: str) -> None:
        """Convierte el diccionario de métricas matemáticas en un PDF formal."""
        pdf = FPDF()
        pdf.add_page()
        
        # --- ENCABEZADO ---
        pdf.set_font("Arial", style='B', size=16)
        pdf.cell(200, 10, txt="Reporte de Evaluación de Inteligencia Artificial Visual", ln=True, align='C')
        
        pdf.set_font("Arial", style='I', size=12)
        pdf.cell(200, 10, txt=f"Identificador del Proyecto: {project_name}", ln=True, align='C')
        pdf.ln(10)
        
        binary_metrics = metrics_data.get("binary_metrics", {})
        
        # --- SECCIÓN 1: MATRIZ DE CONFUSIÓN ---
        pdf.set_font("Arial", style='B', size=14)
        pdf.cell(200, 10, txt="1. Matriz de Confusión (Valores Absolutos)", ln=True)
        pdf.ln(5)
        
        pdf.set_font("Arial", size=12)
        cm = binary_metrics.get("confusion_matrix", {})
        
        pdf.cell(200, 8, txt=f"Verdaderos Positivos (TP) [Aciertos de presencia]: {cm.get('true_positives', 0)}", ln=True)
        pdf.cell(200, 8, txt=f"Verdaderos Negativos (TN) [Aciertos de ausencia]: {cm.get('true_negatives', 0)}", ln=True)
        pdf.cell(200, 8, txt=f"Falsos Positivos (FP) [Alucinaciones del modelo]: {cm.get('false_positives', 0)}", ln=True)
        pdf.cell(200, 8, txt=f"Falsos Negativos (FN) [Ceguera del modelo]: {cm.get('false_negatives', 0)}", ln=True)
        pdf.ln(10)

        # --- SECCIÓN 2: MÉTRICAS DE RENDIMIENTO ---
        pdf.set_font("Arial", style='B', size=14)
        pdf.cell(200, 10, txt="2. Métricas de Rendimiento (0.0 a 1.0)", ln=True)
        pdf.ln(5)
        
        pdf.set_font("Arial", size=12)
        for key, value in binary_metrics.items():
            if key != "confusion_matrix":
                # Limpiamos el nombre técnico (ej: 'f1_score' -> 'F1 Score')
                nombre_formateado = key.replace('_', ' ').title()
                pdf.cell(200, 8, txt=f"{nombre_formateado}: {value}", ln=True)
        
        # --- GUARDADO ---
        pdf.output(output_path)