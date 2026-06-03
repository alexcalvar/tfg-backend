import pytest
from src.utils.config_loader import ConfigLoader

# ==========================================
# FIXTURES AVANZADOS
# ==========================================

@pytest.fixture(autouse=True)
def reset_singleton():
    """
    Se ejecuta automáticamente antes y después de CADA test en este archivo.
    Garantiza que el Singleton se destruya y cada test tenga un entorno limpio.
    """
    ConfigLoader._instance = None
    yield  # Aquí es donde se ejecuta el test
    ConfigLoader._instance = None

@pytest.fixture
def temp_config_file(tmp_path):
    """
    Crea un archivo config.properties temporal y devuelve su ruta.
    `tmp_path` es inyectado mágicamente por pytest.
    """
    config_content = """
[Paths]
datasets_folder = datasets_test
status_file = status_test.json

[Video]
frame_interval = 2.5
max_intents_frame = 3

[Configuracion]
output_parser_type = json
descrip_per_batch = 5
    """
    # Guardamos el archivo en el directorio temporal
    config_file = tmp_path / "config_test.properties"
    config_file.write_text(config_content, encoding="utf-8")
    
    # Devolvemos la ruta absoluta como string
    return str(config_file)


class TestConfigLoader:

    # ==========================================
    # TESTS DE PATRÓN CREACIONAL (Singleton)
    # ==========================================
    
    def test_singleton_pattern(self, temp_config_file):
        """Prueba que múltiples llamadas devuelven la MISMA instancia de memoria."""
        instancia_1 = ConfigLoader(config_path=temp_config_file)
        instancia_2 = ConfigLoader(config_path=temp_config_file)

        # 'is' verifica identidad de memoria (misma dirección), no solo igualdad (==)
        assert instancia_1 is instancia_2  

    def test_archivo_no_encontrado_lanza_error(self):
        """Prueba que instanciar el ConfigLoader con un archivo inexistente falla ruidosamente."""
        with pytest.raises(FileNotFoundError, match="El ConfigLoader no encuentra el archivo"):
            ConfigLoader(config_path="ruta/inventada/que/no/existe.properties")

    # ==========================================
    # TESTS DE LECTURA Y CONVERSIÓN DE TIPOS
    # ==========================================
    
    def test_get_path_lee_strings_correctamente(self, temp_config_file):
        """Prueba que lee cadenas de texto de la sección [Paths]."""
        config = ConfigLoader(config_path=temp_config_file)
        
        assert config.get_path("datasets_folder") == "datasets_test"
        assert config.get_path("status_file") == "status_test.json"

    def test_get_video_realiza_conversion_de_tipos(self, temp_config_file):
        """Prueba que ConfigParser convierte los strings a int y float."""
        config = ConfigLoader(config_path=temp_config_file)
        
        intervalo = config.get_video_float("frame_interval")
        intentos = config.get_video_int("max_intents_frame")
        
        assert isinstance(intervalo, float)
        assert intervalo == 2.5
        
        assert isinstance(intentos, int)
        assert intentos == 3

    def test_get_all_config_as_dict(self, temp_config_file):
        """Prueba que la configuración se serializa bien para exportarla."""
        config = ConfigLoader(config_path=temp_config_file)
        
        diccionario = config.get_all_config_as_dict()
        
        assert "Paths" in diccionario
        assert "Video" in diccionario
        # ConfigParser internamente guarda todo como string en el diccionario final
        assert diccionario["Video"]["max_intents_frame"] == "3"