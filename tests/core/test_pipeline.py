import pytest
import asyncio
import os
from unittest.mock import patch, MagicMock, AsyncMock
from src.core.pipeline import VLMPipeline
from src.data.validators import FrameResults
from src.utils.project_status import ProjectStatus

# ==========================================
# FIXTURES AVANZADOS (Inyección de Mocks)
# ==========================================

@pytest.fixture
def mock_dependencies():
    """Genera un diccionario con todas las dependencias falsas listas para inyectar."""
    deps = {
        "model_instance": MagicMock(),
        "provider_name": "test_provider",
        "frame_provider": MagicMock(),
        "message_strategy": MagicMock(),
        "processing_strategy": MagicMock(),
        "postprocessing_strategy": MagicMock(),
        "base_run_dir": "/ruta/falsa/run_123"
    }
    
    # Configuramos el comportamiento por defecto de algunos mocks
    deps["frame_provider"].get_source_name.return_value = "video_test.mp4"
    deps["frame_provider"].get_expected_frame_count.return_value = 5
    deps["processing_strategy"].load_prompts.return_value = ("System Prompt", "Task Template")
    
    return deps

@pytest.fixture
def pipeline(mock_dependencies):
    """Instancia el Pipeline aislando completamente el sistema de archivos y configuración."""
    with patch('src.core.pipeline.ConfigLoader'), \
         patch('src.core.pipeline.ProjectStatusManager'), \
         patch('src.core.pipeline.ensure_dir'):
         
        pipe = VLMPipeline(**mock_dependencies)
        return pipe

class TestVLMPipeline:

    # ==========================================
    # TESTS SÍNCRONOS (Inicialización y Guardado)
    # ==========================================

    def test_pipeline_initialization(self, pipeline, mock_dependencies):
        """Prueba que el orquestador vincula correctamente el patrón Observer en su constructor."""
        # Verificamos que la estrategia de procesamiento se suscribió al status_manager
        mock_strategy = mock_dependencies["processing_strategy"]
        mock_strategy.attach.assert_called_once_with(pipeline.status_manager)
        
        # Verificamos que se cargaron los prompts
        assert pipeline.system_prompt == "System Prompt"
        assert pipeline.task_template == "Task Template"

    @patch('src.core.pipeline.save_json')
    @patch('src.core.pipeline.datetime')
    def test_save_execution_config(self, mock_datetime, mock_save_json, pipeline):
        """Prueba que el metadato de la ejecución se empaqueta correctamente."""
        # Simulamos una fecha fija para que el test sea determinista
        mock_datetime.datetime.now.return_value.strftime.return_value = "2026-06-01 12:00:00"
        pipeline.config.get_video_float.return_value = 2.0
        pipeline.config.get_video_int.return_value = 5

        pipeline._save_execution_config("video.mp4", "Busca al perro", 100)

        # Verificamos que intentó guardar el JSON
        mock_save_json.assert_called_once()
        
        # Extraemos el diccionario que intentó guardar
        argumentos_llamada = mock_save_json.call_args[0]
        diccionario_guardado = argumentos_llamada[0]
        ruta_guardado = argumentos_llamada[1]
        
        assert ruta_guardado == os.path.join("/ruta/falsa/run_123", "execution_config.json")
        assert diccionario_guardado["inference_parameters"]["total_frames"] == 100
        assert diccionario_guardado["inference_parameters"]["user_query"] == "Busca al perro"

    # ==========================================
    # TEST ASÍNCRONO MAESTRO (El flujo completo)
    # ==========================================

    @pytest.mark.asyncio
    @patch('src.core.pipeline.save_results')
    @patch('src.core.pipeline.VLMPipeline._save_execution_config')
    async def test_process_video_flujo_completo(self, mock_save_config, mock_save_results, pipeline, mock_dependencies):
        """
        Prueba el núcleo asíncrono de la aplicación: Productor, Consumidor y Ordenamiento.
        """
        # 1. PREPARACIÓN DEL PRODUCTOR (frame_provider)
        # Simulamos que la extracción mete dos frames en la cola de forma asíncrona
        async def fake_extract(cola_frames, cancel_event=None):
            # Ojo: Los metemos desordenados (ID 2 primero, ID 1 después)
            await cola_frames.put(MagicMock(frame_id=2))
            await cola_frames.put(MagicMock(frame_id=1))
            
        mock_dependencies["frame_provider"].extract_frames.side_effect = fake_extract

        # 2. PREPARACIÓN DEL CONSUMIDOR (processing_strategy)
        # Simulamos que la estrategia saca los items, genera resultados y hace task_done() para no bloquear
        async def fake_process_queue(processor, prompt, queue, resultados, cancel_event=None):
            while True:
                item = await queue.get()
                if item is None:
                    queue.task_done()
                    break
                # Generamos un resultado falso copiando el frame_id
                resultados.append(FrameResults(frame_id=item.frame_id, detectado=True, descripcion="Exito"))
                queue.task_done()

        mock_dependencies["processing_strategy"].process_queue.side_effect = fake_process_queue

        # ACT: ¡Ejecutamos el flujo entero!
        await pipeline.process_video("Analiza esto")

        # ASSERT: Verificamos la orquestación paso a paso
        
        # A) Limpieza de recursos llamada
        mock_dependencies["frame_provider"].cleanup.assert_called_once()
        
        # B) Configuración guardada
        mock_save_config.assert_called_once()
        
        # C) Verificamos que los resultados se ordenaron correctamente al final
        # Recuerda que el Productor los metió como [ID 2, ID 1]
        args_save_results = mock_save_results.call_args[0]
        lista_resultados = args_save_results[0]
        assert len(lista_resultados) == 2
        assert lista_resultados[0].frame_id == 1  # ¡Se ordenó!
        assert lista_resultados[1].frame_id == 2
        
        # D) Postprocesado ejecutado
        mock_dependencies["postprocessing_strategy"].execute.assert_called_once_with(
            lista_resultados, os.path.join("/ruta/falsa/run_123", "results")
        )
        
        # E) Estado actualizado al final
        pipeline.status_manager.update_status.assert_called_with(ProjectStatus.COMPLETED, "Análisis finalizado con éxito.", 5)