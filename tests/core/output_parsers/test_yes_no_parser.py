import pytest
from src.core.output_parsers.yes_no_parser import YesNoTextParser
from src.data.validators import FrameResults, FramesPath

@pytest.fixture
def parser():
    """Fixture para instanciar el parser una sola vez por test."""
    return YesNoTextParser()

class TestYesNoTextParser:

    # ==========================================
    # TESTS PARA parse() - Un solo frame
    # ==========================================

    def test_parse_texto_positivo_estandar(self, parser):
        """Prueba una respuesta positiva estándar esperada del modelo."""
        texto_ia = "SÍ - Hay un lobo corriendo en la nieve."
        
        resultado = parser.parse(texto_ia, frame_id=1)
        
        assert isinstance(resultado, FrameResults)
        assert resultado.frame_id == 1
        assert resultado.detectado is True
        assert resultado.descripcion == "Hay un lobo corriendo en la nieve."

    def test_parse_texto_negativo_con_ruido_prefijo(self, parser):
        """Prueba que limpia el prefijo 'Frame X:' y detecta la variante inglesa o sin tilde."""
        texto_ia = "Frame 2: NO. No detecto ningún animal."
        
        resultado = parser.parse(texto_ia, frame_id=2)
        
        assert resultado.detectado is False
        assert resultado.descripcion == "No detecto ningún animal."

    def test_parse_detecta_yes_y_true(self, parser):
        """Prueba que el regex acepta las variantes en inglés (YES, TRUE)."""
        texto_ia = "TRUE: I can see the object."
        
        resultado = parser.parse(texto_ia, frame_id=3)
        
        assert resultado.detectado is True
        assert resultado.descripcion == "I can see the object."

    def test_parse_falso_positivo_en_descripcion(self, parser):
        """Prueba de seguridad: la palabra 'sí' dentro del texto NO debe alterar el booleano si empieza por NO."""
        # El parser mira los primeros 15 caracteres. "NO - " cumple la regla de falso.
        texto_ia = "NO - La respuesta a si hay un perro es no."
        
        resultado = parser.parse(texto_ia, frame_id=4)
        
        assert resultado.detectado is False
        # Verificamos que no recorta más de la cuenta
        assert resultado.descripcion == "La respuesta a si hay un perro es no."

    # ==========================================
    # TESTS PARA parse_batch() - Lote de frames
    # ==========================================

    def test_parse_batch_limpio(self, parser):
        """Prueba un lote donde la IA responde exactamente una línea por frame."""
        lote = [
            FramesPath(frame_id=10, frame_path="ruta1.jpg", intentos=1, timestamp_sec=0.5),
            FramesPath(frame_id=11, frame_path="ruta2.jpg", intentos=1, timestamp_sec=1.0)
        ]
        texto_ia = "SÍ - Gato en la ventana\nNO - Ventana vacía"

        resultados = parser.parse_batch(texto_ia, lote)

        assert len(resultados) == 2
        assert resultados[0].frame_id == 10
        assert resultados[0].detectado is True
        assert resultados[1].frame_id == 11
        assert resultados[1].detectado is False
        assert resultados[1].descripcion == "Ventana vacía"

    def test_parse_batch_con_basura_introductoria(self, parser):
        """Prueba la heurística de rescate: La IA mete saludos y texto extra, pero acierta las líneas."""
        lote = [FramesPath(frame_id=20, frame_path="ruta.jpg", intentos=1, timestamp_sec=0.0)]
        
        texto_ia = """
        ¡Hola! Claro, aquí tienes el análisis solicitado:
        
        SÍ - Se detecta movimiento en el fondo.
        
        Espero que te sirva de ayuda.
        """

        resultados = parser.parse_batch(texto_ia, lote)

        assert len(resultados) == 1
        assert resultados[0].detectado is True
        assert resultados[0].descripcion == "Se detecta movimiento en el fondo."

    def test_parse_batch_lineas_insuficientes_lanza_error(self, parser):
        """Prueba que el sistema lanza error si la IA devuelve menos respuestas que imágenes."""
        lote = [
            FramesPath(frame_id=1, frame_path="ruta1.jpg", intentos=1, timestamp_sec=0.0),
            FramesPath(frame_id=2, frame_path="ruta2.jpg", intentos=1, timestamp_sec=0.5)
        ]
        texto_ia = "SÍ - Solo analicé la primera imagen, lo siento."

        # Como solo hay 1 línea útil pero esperamos 2, debe saltar la excepción
        with pytest.raises(ValueError, match="Se esperaban 2 respuestas, pero solo se detectaron 1 líneas"):
            parser.parse_batch(texto_ia, lote)