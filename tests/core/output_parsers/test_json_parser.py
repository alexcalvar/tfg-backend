import pytest
from src.core.output_parsers.json_parser import JsonFrameParser
from src.data.validators import FrameResults, FramesPath

@pytest.fixture
def parser():
    """Fixture de pytest para instanciar el parser una sola vez por test."""
    return JsonFrameParser()

class TestJsonFrameParser:

    # ==========================================
    # TESTS PARA parse() - Un solo frame
    # ==========================================

    def test_parse_json_limpio(self, parser):
        """Prueba que un JSON perfecto se parsea correctamente."""
        # Arrange
        texto_ia = '{"detectado": true, "descripcion": "Hay un perro."}'
        
        # Act
        resultado = parser.parse(texto_ia, frame_id=1)
        
        # Assert
        assert isinstance(resultado, FrameResults)
        assert resultado.frame_id == 1
        assert resultado.detectado is True
        assert resultado.descripcion == "Hay un perro."

    def test_parse_json_con_markdown(self, parser):
        """Prueba la heurística de limpieza cuando la IA responde con bloques markdown."""
        # Arrange
        texto_ia = '''```json
        {"detectado": "True", "descripcion": "Markdown detectado"}
        ```'''
        
        # Act
        resultado = parser.parse(texto_ia, frame_id=2)
        
        # Assert
        assert resultado.detectado is True # Ojo: debe normalizar el string "True" a booleano True
        assert resultado.descripcion == "Markdown detectado"

    def test_parse_json_invalido_lanza_error(self, parser):
        """Prueba que un texto que no es JSON lanza la excepción esperada."""
        # Arrange
        texto_ia = "Hola, soy una IA y no quiero responder en JSON."
        
        # Act & Assert
        with pytest.raises(ValueError, match="Error decodificando JSON"):
            parser.parse(texto_ia, frame_id=3)

    # ==========================================
    # TESTS PARA parse_batch() - Lote de frames
    # ==========================================

    def test_parse_batch_lista_correcta(self, parser):
        """Prueba un lote con una lista de respuestas perfectas."""
        # Arrange
        lote = [
            FramesPath(frame_id=10, frame_path="ruta1.jpg", intentos=1, timestamp_sec=0.5),
            FramesPath(frame_id=11, frame_path="ruta2.jpg", intentos=1, timestamp_sec=1.0)
        ]
        texto_ia = '''[
            {"detectado": false, "descripcion": "Nada aquí"},
            {"detectado": true, "descripcion": "Gato detectado"}
        ]'''

        # Act
        resultados = parser.parse_batch(texto_ia, lote)

        # Assert
        assert len(resultados) == 2
        assert resultados[0].frame_id == 10
        assert resultados[0].detectado is False
        assert resultados[1].frame_id == 11
        assert resultados[1].detectado is True
        assert resultados[1].descripcion == "Gato detectado"

    def test_parse_batch_salva_estructura_diccionario_unico(self, parser):
        """Prueba la heurística: la IA devuelve 1 solo dict para un lote de 1 imagen."""
        # Arrange
        lote = [FramesPath(frame_id=20, frame_path="ruta.jpg", intentos=1, timestamp_sec=0.0)]
        texto_ia = '{"detectado": true, "descripcion": "Solo un frame"}' # Faltan los corchetes de lista []

        # Act
        resultados = parser.parse_batch(texto_ia, lote)

        # Assert
        assert len(resultados) == 1
        assert resultados[0].detectado is True

    def test_parse_batch_desajuste_de_tamaño_lanza_error(self, parser):
        """Prueba que falla si la IA devuelve menos resultados que frames en el lote."""
        # Arrange
        lote = [
            FramesPath(frame_id=1, frame_path="ruta1.jpg", intentos=1, timestamp_sec=0.0),
            FramesPath(frame_id=2, frame_path="ruta2.jpg", intentos=1, timestamp_sec=0.5)
        ]
        texto_ia = '[{"detectado": true, "descripcion": "Me comí un frame"}]'

        # Act & Assert
        with pytest.raises(ValueError, match="Desajuste: El modelo devolvió 1 resultados"):
            parser.parse_batch(texto_ia, lote)