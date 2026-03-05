class BinaryMetricsCalculator:
    def __init__(self):
        # Matriz de Confusión base
        self.tp = 0 # True Positives (IA dice SÍ, Realidad dice SÍ)
        self.tn = 0 # True Negatives (IA dice NO, Realidad dice NO)
        self.fp = 0 # False Positives (IA dice SÍ, Realidad dice NO - Alucinación)
        self.fn = 0 # False Negatives (IA dice NO, Realidad dice SÍ - Ceguera)

    def compute_matrix(self, evaluaciones: list) -> None:
        for eval in evaluaciones:
            if eval.ground_truth_detectado and eval.modelo_detectado:
                self.tp += 1
            elif not eval.ground_truth_detectado and not eval.modelo_detectado:
                self.tn += 1
            elif not eval.ground_truth_detectado and eval.modelo_detectado:
                self.fp += 1
            elif eval.ground_truth_detectado and not eval.modelo_detectado:
                self.fn += 1

    def calculate_all_metrics(self) -> dict:
        # Prevenimos divisiones por cero con variables de seguridad
        total_positives = self.tp + self.fn
        total_negatives = self.tn + self.fp
        total_predictions = self.tp + self.tn + self.fp + self.fn

        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        recall = self.tp / total_positives if total_positives > 0 else 0.0
        specificity = self.tn / total_negatives if total_negatives > 0 else 0.0

        return {
            "binary_metrics": {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(2 * (precision * recall) / (precision + recall), 4) if (precision + recall) > 0 else 0.0,
                "accuracy": round((self.tp + self.tn) / total_predictions, 4) if total_predictions > 0 else 0.0,
                "sensitivity": round(recall, 4),
                "specificity": round(specificity, 4),
                "youden_index": round(recall + specificity - 1, 4),
                "confusion_matrix": {
                    "true_positives": self.tp,
                    "true_negatives": self.tn,
                    "false_positives": self.fp,
                    "false_negatives": self.fn
                }
            }
        }