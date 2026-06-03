import pytest
import asyncio
import os
import cv2
from unittest.mock import patch, MagicMock, call
from src.core.frame_providers.video_loader import VideoLoader
from src.data.validators import FramesPath

# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def mock_config():
    """Parchea el ConfigLoader para evitar que lea de disco y dictar la configuración deseada."""
    with patch('src.core.frame_providers.video_loader.ConfigLoader') as MockConfig:
        config = MockConfig.return_value
        # Simulamos que la configuración exige 3 intentos y redimensión a 640x360
        config.get_video_int.side_effect = lambda key: {
            "max_intents_frame": 3,
            "resize_width": 640,
            "resize_height": 360
        }.get(key, 0)
        yield config

@pytest.fixture
def loader(mock_config):
    """Instancia el VideoLoader interceptando ensure_dir para que no cree carpetas reales."""
    with patch('src.core.frame_providers.video_loader.ensure_dir'):
        # Configuramos 1 frame por segundo (interval = 1.0)
        return VideoLoader("ruta/falsa/video_test.mp4", "ruta/falsa/salida", interval=1.0)

class TestVideoLoader:

    # ==========================================
    # TESTS DE CÁLCULO (get_expected_frame_count)
    # ==========================================

    @patch('src.core.frame_providers.video_loader.cv2.VideoCapture')
    def test_get_expected_frame_count(self, mock_video_capture, loader):
        """Prueba la matemática de estimación de frames sin extraerlos."""
        # Creamos un objeto falso que simulará ser el vídeo abierto
        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True

        # Simulamos que el vídeo va a 30 FPS y tiene 90 frames en total (3 segundos)
        def mock_get(prop_id):
            if prop_id == cv2.CAP_PROP_FPS: return 30.0
            if prop_id == cv2.CAP_PROP_FRAME_COUNT: return 90.0
            return 0.0
        
        mock_cap.get.side_effect = mock_get

        # Act
        total_estimado = loader.get_expected_frame_count()

        # Assert: Si el intervalo es 1.0s, avanzamos de 30 en 30 frames. 
        # En 90 frames, deberíamos extraer exactamente 3 (90 / 30).
        assert total_estimado == 3

    # ==========================================
    # TESTS DE INFRAESTRUCTURA Y LIMPIEZA
    # ==========================================

    def test_get_source_name(self, loader):
        """Verifica que extrae bien el nombre del archivo de la ruta."""
        assert loader.get_source_name() == "video_test.mp4"

    @patch('src.core.frame_providers.video_loader.os.remove')
    @patch('src.core.frame_providers.video_loader.os.path.exists', return_value=True)
    @patch('src.core.frame_providers.video_loader.os.path.isdir', return_value=True)
    @patch('src.core.frame_providers.video_loader.os.listdir', return_value=[]) # Carpeta vacía
    @patch('src.core.frame_providers.video_loader.os.rmdir')
    def test_cleanup_exitoso(self, mock_rmdir, mock_listdir, mock_isdir, mock_exists, mock_remove, loader):
        """Prueba que borra el vídeo original y su carpeta si queda vacía."""
        # Act
        loader.cleanup()
        
        # Assert
        mock_remove.assert_called_once_with("ruta/falsa/video_test.mp4")
        mock_rmdir.assert_called_once_with("ruta/falsa")

    # ==========================================
    # TEST DEL BUCLE PRINCIPAL (ASÍNCRONO)
    # ==========================================

    @pytest.mark.asyncio
    @patch('src.core.frame_providers.video_loader.cv2.imwrite')
    @patch('src.core.frame_providers.video_loader.cv2.resize')
    @patch('src.core.frame_providers.video_loader.cv2.VideoCapture')
    async def test_extract_frames_exitoso(self, mock_video_capture, mock_resize, mock_imwrite, loader):
        """Prueba el bucle de extracción, redimensión y encolado."""
        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True

        # Simulamos los metadatos de OpenCV
        def mock_get(prop_id):
            if prop_id == cv2.CAP_PROP_FPS: return 30.0
            if prop_id == cv2.CAP_PROP_FRAME_COUNT: return 90.0
            if prop_id == cv2.CAP_PROP_POS_MSEC: return 1500.0 # Simulamos 1.5 seg
            return 0.0
        mock_cap.get.side_effect = mock_get

        # Simulamos la lectura de frames: Lee 2 frames con éxito y el 3º falla (fin del vídeo)
        mock_cap.read.side_effect = [(True, "raw_frame_1"), (True, "raw_frame_2"), (False, None)]
        
        # Simulamos que OpenCV redimensiona y devuelve una matriz falsa
        mock_resize.return_value = "frame_redimensionado_fake"

        cola = asyncio.Queue()

        # Act
        await loader.extract_frames(cola)

        # Assert: Debe haber 2 frames en la cola
        assert cola.qsize() == 2
        
        # Comprobamos el contenido del primer frame encolado
        frame_extraido = await cola.get()
        assert isinstance(frame_extraido, FramesPath)
        assert frame_extraido.frame_id == 0
        assert frame_extraido.intentos == 3  # Valor inyectado por mock_config
        assert frame_extraido.timestamp_sec == 1.5

        # Verificamos que OpenCV intentó guardar físicamente 2 archivos
        assert mock_imwrite.call_count == 2
        
        # Verificamos qué ruta le mandó a guardar OpenCV
        ruta_guardado = os.path.join("ruta/falsa/salida", "frame_0.jpg")
        mock_imwrite.assert_any_call(ruta_guardado, "frame_redimensionado_fake")
        
        # Verificamos que se avanzó el vídeo usando cap.set
        mock_cap.set.assert_called_with(cv2.CAP_PROP_POS_FRAMES, 60) # n_frame inicial (0) + step (30)