import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
import os

from src.api.routes import endpoints
from src.utils.project_status import ProjectStatus

# ==========================================
# SETUP DE LA APLICACIÓN DE TEST
# ==========================================

# Creamos una mini-aplicación FastAPI solo para montar tus rutas
app = FastAPI()
app.include_router(endpoints)

# Instanciamos el cliente de pruebas
client = TestClient(app)

class TestAPIRoutes:

    # ==========================================
    # TESTS DEL ENDPOINT POST (Análisis de Eventos)
    # ==========================================

    @patch('src.api.routes.AnalysisOrchestrator')
    def test_post_analyze_video_exitoso(self, mock_orchestrator_class):
        """Prueba que el endpoint acepta el vídeo y lanza la tarea en segundo plano."""
        # 1. Preparar el Mock del Orquestador
        mock_orchestrator_instance = MagicMock()
        mock_pipeline = MagicMock() 
        # Simular que el process_video es asíncrono
        mock_pipeline.process_video = AsyncMock() 
        
        # setup_video_pipeline es asíncrono, devolvemos un coroutine simulado
        mock_orchestrator_instance.setup_video_pipeline = AsyncMock(return_value=(mock_pipeline, "project_123"))
        mock_orchestrator_class.return_value = mock_orchestrator_instance

        # 2. Preparar los datos del formulario HTTP (multipart/form-data)
        datos_formulario = {
            "interval_time": "2.0",
            "user_prompt": "Busca perros",
            "vlm_provider": "google",
            "vlm_model_name": "gemini-1.5-flash",
            "processing_mode": "batch_strategy",
            "apply_alg": "true"
        }
        
        # Creamos un archivo falso en memoria simulando un MP4
        archivo_falso = ("video_test.mp4", b"contenido_binario_falso", "video/mp4")

        # 3. ACT: Enviar la petición POST
        response = client.post(
            "/api/v1/events",
            data=datos_formulario,
            files={"video": archivo_falso}
        )

        # 4. ASSERT: Verificamos la respuesta HTTP
        assert response.status_code == 202 # Accepted (código correcto para tareas en background)
        json_resp = response.json()
        
        assert json_resp["success"] is True
        assert json_resp["data"]["project_id"] == "project_123"
        assert json_resp["data"]["status"] == "processing"

    def test_post_analyze_video_rechaza_archivos_no_video(self):
        """Prueba la validación de seguridad del content-type."""
        datos_formulario = {
            "interval_time": "2.0",
            "user_prompt": "x", "vlm_provider": "y", "vlm_model_name": "z",
            "processing_mode": "w", "apply_alg": "true"
        }
        # Intentamos colar un archivo ejecutable como si fuera un vídeo
        archivo_falso = ("virus.exe", b"malware", "application/x-msdownload")

        response = client.post("/api/v1/events", data=datos_formulario, files={"video": archivo_falso})

        # Debe saltar el HTTPException de la línea 32 de tu código
        assert response.status_code == 400
        assert "El archivo debe ser un formato de vídeo válido" in response.json()["detail"]

    # ==========================================
    # TESTS DEL ENDPOINT GET (Estado)
    # ==========================================

    @patch('src.api.routes.config')
    @patch('src.api.routes.os.path.exists', return_value=True)
    @patch('src.api.routes.load_json')
    def test_get_project_status_exitoso(self, mock_load_json, mock_exists, mock_config):
        """Prueba que el endpoint devuelve el JSON de estado correctamente."""
        # Simulamos la ruta
        mock_config.get_status_file_path.return_value = "/ruta/falsa/status.json"
        
        # Simulamos lo que leería del archivo status.json
        mock_load_json.return_value = {
            "state": "analyzing_frames",
            "progress": {"current_frame": 10, "total_frames": 100},
            "last_updated": "2026-06-03 10:00:00"
        }

        response = client.get("/api/v1/project_123/status")

        assert response.status_code == 200
        json_resp = response.json()
        assert json_resp["success"] is True
        assert json_resp["data"]["state"] == "analyzing_frames"
        assert json_resp["data"]["progress"]["current_frame"] == 10

    # ==========================================
    # TESTS DEL ENDPOINT DELETE (Seguridad)
    # ==========================================

    @patch('src.api.routes.config')
    @patch('src.api.routes.os.path.exists', return_value=True)
    @patch('src.api.routes.shutil.rmtree')
    def test_delete_project_exitoso(self, mock_rmtree, mock_exists, mock_config):
        """Prueba que se borra la carpeta del proyecto si existe y es válido."""
        mock_config.get_path.return_value = "/var/www/projects"
        
        response = client.delete("/api/v1/project_123")

        assert response.status_code == 200
        # Verificamos que se llamó a la función destructiva de borrado con la ruta correcta
        mock_rmtree.assert_called_once_with(os.path.join("/var/www/projects", "project_123"))

    def test_delete_project_bloquea_path_traversal_interno(self):
        # Este ID es un formato que FastAPI sí enruta, pero tu lógica de seguridad debe atrapar
        response = client.delete("/api/v1/project_.._etc")
        
        # Ahora sí, debería saltar tu validación de la línea 249 de routes.py
        assert response.status_code == 400
        assert "ID de proyecto inválido" in response.json()["detail"]
        
    def test_delete_project_bloquea_prefijo_invalido(self):
        """Prueba de validación: el id debe empezar por 'project_'."""
        response = client.delete("/api/v1/mi_carpeta_secreta")

        assert response.status_code == 400
        assert "Formato de ID de proyecto incorrecto" in response.json()["detail"]