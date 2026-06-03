import pytest
import asyncio
from unittest.mock import patch, MagicMock
from src.core.processing_strategies.temporal_strategy import TemporalStrategy
from src.core.output_parsers.base_parser import BaseFrameParser
from src.data.validators import FramesPath, FrameResults

# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def mock_parser():
    return MagicMock(spec=BaseFrameParser)

@pytest.fixture
def strategy(mock_parser):
    """Instancia la estrategia mockeando el ConfigLoader."""
    with patch('src.core.processing_strategies.base_strategy.ConfigLoader'):
        strat = TemporalStrategy(parser=mock_parser)
        # Simulamos que la configuración dicta un máximo de 2 intentos por frame
        strat.config.get_video_int.return_value = 2 
        return strat

class TestTemporalStrategy:

    # ==========================================
    # TESTS DE FORMATEO Y EXTRACCIÓN
    # ==========================================

    def test_build_model_request(self, strategy):
        """Prueba que el formato temporal manda 1 prompt y N imágenes."""
        prompt = "Analiza el movimiento"
        buffer = [
            FramesPath(frame_id=1, frame_path="r1.jpg", intentos=1, timestamp_sec=0.5),
            FramesPath(frame_id=2, frame_path="r2.jpg", intentos=1, timestamp_sec=1.0)
        ]
        
        peticion = strategy._build_model_request(prompt, buffer)
        
        # 1 texto de prompt + 2 imágenes = 3 elementos
        assert len(peticion) == 3
        assert peticion[0] == {"type": "text", "content": prompt}
        assert peticion[1] == {"type": "image", "content": buffer[0]}
        assert peticion[2] == {"type": "image", "content": buffer[1]}

    @pytest.mark.asyncio
    async def test_extract_batch_con_fin_de_cola(self, strategy):
        """Prueba que al recibir None (fin del vídeo), lo vuelve a meter en la cola para no romper el join()."""
        cola = asyncio.Queue()
        await cola.put(FramesPath(frame_id=1, frame_path="r1.jpg", intentos=1, timestamp_sec=0.0))
        await cola.put(None) # Flag de fin
        
        # Pedimos 2 frames, pero la cola tiene 1 frame y un None
        batch = await strategy._extract_batch(cola, batch_size=2)
        
        assert len(batch) == 1
        assert batch[0].frame_id == 1
        # Verificamos que el None volvió a entrar a la cola para avisar a otras estrategias/orquestador
        assert await cola.get() is None

    # ==========================================
    # TESTS DE LÓGICA DE REINTENTOS (SÍNCRONA EN ESTE CASO)
    # ==========================================

    @pytest.mark.asyncio
    async def test_process_batch_interno_reintento_y_fallo(self, strategy, mock_parser):
        """Prueba que agota los intentos y genera un Fallback de error si el modelo falla siempre."""
        resultados = []
        target_frame = FramesPath(frame_id=10, frame_path="r10.jpg", intentos=1, timestamp_sec=5.0)
        buffer = [target_frame]
        
        processor_mock = MagicMock()
        # Forzamos que la IA lance error cada vez que se le llame
        processor_mock.process_layout.side_effect = Exception("Caída del servidor IA")
        
        await strategy._process_batch_interno(processor_mock, "Prompt", buffer, target_frame, resultados)
        
        # Al fallar, debería haber llamado al procesador 2 veces (lo definimos en el fixture)
        assert processor_mock.process_layout.call_count == 2
        # Como agotó los intentos, debe haber generado un resultado de error
        assert len(resultados) == 1
        assert resultados[0].detectado is False
        assert "Error: Fallo de análisis temporal tras 2 intentos" in resultados[0].descripcion

    # ==========================================
    # TEST MAESTRO: LÓGICA DE VENTANA DESLIZANTE
    # ==========================================

    @pytest.mark.asyncio
    async def test_flujo_completo_ventanas_temporales(self, strategy):
        """Prueba que las fases (Inicial, Media, Final) evalúan los índices correctos."""
        cola = asyncio.Queue()
        # Simulamos un minivídeo de 4 frames
        for i in range(1, 5):
            await cola.put(FramesPath(frame_id=i, frame_path=f"r{i}.jpg", intentos=1, timestamp_sec=i*1.0))
        await cola.put(None) # Flag de fin

        resultados = []
        processor_mock = MagicMock()

        # Listas para guardar las "fotocopias" del estado en cada paso
        buffers_historico = []
        indices_historico = []

        with patch.object(strategy, '_evaluar_ventana') as mock_evaluar:
            
            async def fake_evaluar(processor, user_prompt, frame_buffer, target_index, status_message, queue, resultados):
                # ¡AQUÍ ESTÁ LA MAGIA! Extraemos los IDs y creamos una lista NUEVA en memoria
                buffers_historico.append([f.frame_id for f in frame_buffer])
                indices_historico.append(target_index)
                queue.task_done()
            
            mock_evaluar.side_effect = fake_evaluar

            # Ejecutamos todo el flujo principal
            await strategy.process_queue(processor_mock, "Prompt", cola, resultados)

            # Comprobamos la magia de tu algoritmo de ventana deslizante:
            assert mock_evaluar.call_count == 4
            
            # Ahora verificamos contra nuestras fotocopias, que son inmunes a los cambios posteriores
            # LLAMADA 1 (Fase Inicial): Debe tener frame 1 y 2. El objetivo es el 1 (índice 0)
            assert buffers_historico[0] == [1, 2]
            assert indices_historico[0] == 0
            
            # LLAMADA 2 (Fase Media): Debe tener frame 1, 2 y 3. Objetivo es el 2 (índice 1)
            assert buffers_historico[1] == [1, 2, 3]
            assert indices_historico[1] == 1
            
            # LLAMADA 3 (Fase Media): Debe tener frame 2, 3 y 4. Objetivo es el 3 (índice 1)
            assert buffers_historico[2] == [2, 3, 4]
            assert indices_historico[2] == 1
            
            # LLAMADA 4 (Fase Final): Debe tener frame 3 y 4. Objetivo es el 4 (índice -1)
            assert buffers_historico[3] == [3, 4]
            assert indices_historico[3] == -1