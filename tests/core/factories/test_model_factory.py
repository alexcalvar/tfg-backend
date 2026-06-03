import pytest
from unittest.mock import patch, MagicMock
from src.core.factories.model_factory import ModelFactory
from src.core.message_strategies.message_builders import CloudMessageBuilder, LocalMessageBuilder

# ==========================================
# FIXTURE CON MOCKING COMPLETO DE ENTORNO
# ==========================================

@pytest.fixture
def factory_mocked():
    """
    Instancia el ModelFactory interceptando la lectura del archivo JSON
    para inyectarle una configuración controlada que no dependa del disco duro.
    """
    fake_models_config = {
        "vlms": {
            "google": {
                "gemini_test": {"model_string": "gemini-1.5-flash", "temperature": 0.0}
            },
            "ollama": {
                "llama_vision_test": {"model_string": "llama3.2-vision", "temperature": 0.0}
            }
        },
        "llms": {
            "groq": {
                "mixtral_test": {"model_string": "mixtral-8x7b", "temperature": 0.0}
            }
        }
    }
    
    # Parcheamos load_json para que devuelva nuestro diccionario falso
    with patch('src.core.factories.model_factory.load_json', return_value=fake_models_config), \
         patch('src.core.factories.model_factory.ConfigLoader'): 
         # Parcheamos ConfigLoader para que no intente leer config.properties
        
        factory = ModelFactory()
        yield factory

class TestModelFactory:

    # ==========================================
    # TESTS PARA MODELOS VISUALES (VLMs)
    # ==========================================

    @patch('os.getenv', return_value="FAKE_GOOGLE_KEY")
    @patch('src.core.factories.model_factory.ChatGoogleGenerativeAI')
    def test_load_vlm_google_exitoso(self, mock_chat_google, mock_getenv, factory_mocked):
        """Prueba que el Factory construye un VLM de la nube y le asigna CloudMessageBuilder."""
        
        # Act
        modelo, estrategia = factory_mocked.load_vlm("google", "gemini_test")
        
        # Assert
        assert isinstance(estrategia, CloudMessageBuilder)
        # Verificamos que LangChain fue llamado con los parámetros exactos del JSON falso y la API Key
        mock_chat_google.assert_called_once_with(
            model="gemini-1.5-flash", 
            google_api_key="FAKE_GOOGLE_KEY", 
            temperature=0.0
        )

    @patch('src.core.factories.model_factory.ChatOllama')
    def test_load_vlm_ollama_local(self, mock_chat_ollama, factory_mocked):
        """Prueba que el Factory construye un VLM local y le asigna LocalMessageBuilder."""
        
        # Act
        modelo, estrategia = factory_mocked.load_vlm("ollama", "llama_vision_test")
        
        # Assert
        assert isinstance(estrategia, LocalMessageBuilder)
        mock_chat_ollama.assert_called_once_with(
            model="llama3.2-vision", 
            temperature=0.0
        )

    # ==========================================
    # TESTS PARA MODELOS DE TEXTO (LLMs)
    # ==========================================

    @patch('os.getenv', return_value="FAKE_GROQ_KEY")
    @patch('src.core.factories.model_factory.ChatGroq')
    def test_load_llm_groq(self, mock_chat_groq, mock_getenv, factory_mocked):
        """Prueba la instanciación de un LLM de Groq."""
        
        modelo = factory_mocked.load_llm("groq", "mixtral_test")
        
        mock_chat_groq.assert_called_once_with(
            model="mixtral-8x7b", 
            temperature=0.0, 
            api_key="FAKE_GROQ_KEY"
        )

    # ==========================================
    # TESTS DE SEGURIDAD Y MANEJO DE ERRORES
    # ==========================================

    @patch('os.getenv', return_value=None)
    def test_vlm_google_lanza_error_sin_api_key(self, mock_getenv, factory_mocked):
        """Prueba que el sistema se protege si falta la clave en el .env."""
        with pytest.raises(ValueError, match="Falta GOOGLE_API_KEY en el archivo .env"):
            factory_mocked.load_vlm("google", "gemini_test")

    def test_proveedor_no_soportado_lanza_error(self, factory_mocked):
        """Prueba el comportamiento ante un string de proveedor inválido."""
        with pytest.raises(ValueError, match="no está soportado"):
            factory_mocked.load_vlm("anthropic_inventado", "modelo_x")

    def test_modelo_no_configurado_lanza_error(self, factory_mocked):
        """Prueba el comportamiento si se pide un modelo que no está en el JSON."""
        with pytest.raises(ValueError, match="No se encontró configuración para el modelo"):
            factory_mocked.load_vlm("google", "modelo_inventado_que_no_existe")