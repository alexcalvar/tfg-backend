import pytest
import asyncio
from unittest.mock import patch, MagicMock
from src.core.processing_strategies.batch_strategy import BatchStrategy
from src.core.output_parsers.base_parser import BaseFrameParser
from src.data.validators import FramesPath, FrameResults

# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def mock_parser():
    """Crea un parser falso (Mock) para inyectarlo en la estrategia."""
    parser = MagicMock(spec=BaseFrameParser)
    return parser

@pytest.fixture
def strategy(mock_parser):
    """Instancia la estrategia bloqueando la lectura de disco de ConfigLoader."""
    with patch('src.core.processing_strategies.base_strategy.ConfigLoader'):
        return BatchStrategy(parser=mock_parser)

class TestBatchStrategy:

    # ==========================================
    # TESTS SÍNCRONOS (Formateo y Auxiliares)
    # ==========================================

    def test_build_model_request(self, strategy):
        """Prueba que los frames y el prompt se ensamblan en el formato correcto para el modelo."""
        prompt = "Describe este frame"
        lote = [
            FramesPath(frame_id=1, frame_path="ruta1.jpg", intentos=1, timestamp_sec=0.5),
            FramesPath(frame_id=2, frame_path="ruta2.jpg", intentos=1, timestamp_sec=1.0)
        ]
        
        peticion = strategy._build_model_request(prompt, lote)
        
        assert len(peticion) == 4
        assert peticion[0] == {"type": "text", "content": prompt}
        assert peticion[1] == {"type": "image", "content": lote[0]}
        assert peticion[2] == {"type": "text", "content": prompt}
        assert peticion[3] == {"type": "image", "content": lote[1]}

    # ==========================================
    # TESTS ASÍNCRONOS (Colas y Lógica de Reintento)
    # ==========================================

    @pytest.mark.asyncio
    async def test_extract_batch_vacia_cola_correctamente(self, strategy):
        """Prueba que extrae exactamente N elementos de la cola."""
        cola = asyncio.Queue()
        await cola.put(FramesPath(frame_id=1, frame_path="r1.jpg", intentos=1, timestamp_sec=0.0))
        await cola.put(FramesPath(frame_id=2, frame_path="r2.jpg", intentos=1, timestamp_sec=0.5))
        await cola.put(FramesPath(frame_id=3, frame_path="r3.jpg", intentos=1, timestamp_sec=1.0))
        
        batch = await strategy._extract_batch(cola, batch_size=2)
        
        assert len(batch) == 2
        assert batch[0].frame_id == 1
        assert batch[1].frame_id == 2
        assert cola.qsize() == 1 

    @pytest.mark.asyncio
    async def test_handle_batch_failure_reencola_con_menos_intentos(self, strategy):
        """Prueba el sistema de tolerancia a fallos: Si hay intentos, vuelve a la cola."""
        cola = asyncio.Queue()
        resultados = []
        lote = [FramesPath(frame_id=10, frame_path="r10.jpg", intentos=3, timestamp_sec=5.0)]
        
        await strategy._handle_batch_failure(lote, resultados, cola, "Fallo de red")
        
        assert cola.qsize() == 1 
        frame_reencolado = await cola.get()
        assert frame_reencolado.intentos == 2 
        assert frame_reencolado.timestamp_sec == 5.0 # Verificamos que no perdió el timestamp
        assert len(resultados) == 0 

    @pytest.mark.asyncio
    async def test_handle_batch_failure_agota_intentos_y_falla(self, strategy):
        """Prueba que si se acaban los intentos, genera un FrameResults de error."""
        cola = asyncio.Queue()
        resultados = []
        lote = [FramesPath(frame_id=99, frame_path="r99.jpg", intentos=1, timestamp_sec=20.0)]
        
        await strategy._handle_batch_failure(lote, resultados, cola, "IA Caída")
        
        assert cola.qsize() == 0 
        assert len(resultados) == 1 
        assert resultados[0].detectado is False
        assert "Error Crítico: IA Caída" in resultados[0].descripcion

    # ==========================================
    # TEST DE INTEGRACIÓN AISLADA (El núcleo)
    # ==========================================

    @pytest.mark.asyncio
    async def test_process_batch_interno_flujo_exitoso(self, strategy, mock_parser):
        """Prueba que el procesado interno llama al modelo y al parser correctamente."""
        cola = asyncio.Queue()
        resultados = []
        lote = [FramesPath(frame_id=1, frame_path="r1.jpg", intentos=1, timestamp_sec=0.0)]
        
        processor_mock = MagicMock()
        processor_mock.process_layout.return_value = "Respuesta simulada de la IA"
        
        mock_parser.parse_batch.return_value = [
            FrameResults(frame_id=1, detectado=True, descripcion="Exito")
        ]
        
        await strategy._process_batch_interno(processor_mock, "Prompt", lote, cola, resultados)
        
        processor_mock.process_layout.assert_called_once()
        mock_parser.parse_batch.assert_called_once_with("Respuesta simulada de la IA", lote)
        assert len(resultados) == 1
        assert resultados[0].detectado is True