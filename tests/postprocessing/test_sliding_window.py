import pytest
from unittest.mock import patch
from src.postprocessing.postprocessing_algorithms.sliding_window import SlidingWindowNormalizer
from src.data.validators import FrameResults

# ==========================================
# HELPER (Patrón Factory para Tests)
# ==========================================
def generar_frames(valores: list[bool]) -> list[FrameResults]:
    """Genera rápidamente una lista de FrameResults basada en una lista de booleanos."""
    return [
        FrameResults(frame_id=i, detectado=val, descripcion=f"Frame {i}")
        for i, val in enumerate(valores)
    ]

# ==========================================
# FIXTURE CON MOCKING
# ==========================================
@pytest.fixture
def normalizer():
    """
    Instancia el algoritmo parcheando el ConfigLoader para no depender
    del archivo config.properties en el disco duro.
    """
    with patch('src.utils.config_loader.ConfigLoader.get_video_float', return_value=2.0):
        # Configuramos un intervalo ficticio de 2.0 segundos por frame
        norm = SlidingWindowNormalizer(apply_alg=True)
        # Forzamos la ventana a 5 (t-2, t-1, t, t+1, t+2) que es tu valor por defecto
        norm.window_size = 5 
        return norm

class TestSlidingWindowNormalizer:

    # ==========================================
    # 1. TESTS MATEMÁTICOS (Filtro de Ventana)
    # ==========================================

    def test_eliminar_falso_positivo_aislado(self, normalizer):
        """Prueba que un True rodeado de Falses se convierte en False."""
        # Arrange: 0, 1, [2], 3, 4
        frames_sucios = generar_frames([False, False, True, False, False])
        
        # Act
        frames_limpios = normalizer._apply_algoritm(frames_sucios)
        
        # Assert
        # En la ventana alrededor del índice 2, hay 1 True y 4 Falses. Debe aplanarlo.
        assert frames_limpios[2].detectado is False

    def test_eliminar_falso_negativo_aislado(self, normalizer):
        """Prueba que un False (ceguera del modelo) en medio de un evento se corrige a True."""
        # Arrange
        frames_sucios = generar_frames([True, True, False, True, True])
        
        # Act
        frames_limpios = normalizer._apply_algoritm(frames_sucios)
        
        # Assert
        # En la ventana alrededor del 2, hay 4 Trues y 1 False. Debe rellenarlo.
        assert frames_limpios[2].detectado is True

    def test_bordes_de_la_ventana_respetados(self, normalizer):
        """Prueba que el algoritmo no falla al principio o al final de la lista (Out of Bounds)."""
        # Arrange: Un True en el primer frame. Ventana [0:3] -> 1 True, 2 Falses. Debe dar False.
        frames_sucios = generar_frames([True, False, False, False, False])
        
        # Act
        frames_limpios = normalizer._apply_algoritm(frames_sucios)
        
        # Assert
        assert frames_limpios[0].detectado is False

    # ==========================================
    # 2. TESTS DE AGRUPACIÓN (Extracción de Intervalos)
    # ==========================================

    def test_extract_intervals_evento_unico(self, normalizer):
        """Prueba que una racha de Trues se empaqueta en un solo EventInterval."""
        # Arrange: Racha de Trues del frame 2 al 4.
        frames_limpios = generar_frames([False, False, True, True, True, False])
        
        # Act
        eventos = normalizer._extract_intervals(frames_limpios)
        
        # Assert
        assert len(eventos) == 1
        assert eventos[0].event_id == 0
        assert eventos[0].start_frame == 2
        assert eventos[0].end_frame == 4
        # Hemos mockeado interval_time a 2.0s
        assert eventos[0].start_timestamp == 4.0  # 2 * 2.0
        assert eventos[0].end_timestamp == 8.0    # 4 * 2.0

    def test_extract_intervals_multiples_eventos(self, normalizer):
        """Prueba que separa eventos distintos correctamente."""
        # Arrange: Evento 1 (id 1, 2) y Evento 2 (id 5)
        frames_limpios = generar_frames([False, True, True, False, False, True, False])
        
        # Act
        eventos = normalizer._extract_intervals(frames_limpios)
        
        # Assert
        assert len(eventos) == 2
        assert eventos[0].start_frame == 1
        assert eventos[0].end_frame == 2
        assert eventos[1].start_frame == 5
        assert eventos[1].end_frame == 5

    def test_extract_intervals_cierre_seguridad(self, normalizer):
        """Prueba el 'cierre de seguridad' si el vídeo termina de golpe durante un evento."""
        # Arrange: Termina en True (el evento se queda abierto al acabar el bucle)
        frames_limpios = generar_frames([False, True, True, True])
        
        # Act
        eventos = normalizer._extract_intervals(frames_limpios)
        
        # Assert
        assert len(eventos) == 1
        assert eventos[0].start_frame == 1
        assert eventos[0].end_frame == 3