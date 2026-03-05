import configparser
import os

from utils.file_utils import save_json 

class ConfigLoader:
    _instance = None

    # nombre del archivo por defecto
    def __new__(cls, config_path="config.properties"):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._instance._initialize(config_path)
        return cls._instance

    def _initialize(self, config_path):
        self.config = configparser.ConfigParser()
        
        # calcular la ruta absoluta de forma dinámica.
        ruta_absoluta = os.path.abspath(config_path)
        
        if not os.path.exists(ruta_absoluta):
            raise FileNotFoundError(f"[ERROR] El ConfigLoader no encuentra el archivo en: {ruta_absoluta}")
        
        self.config.read(ruta_absoluta, encoding="utf-8")
        print(f"  Configuración cargada con éxito desde: {config_path}")
    
    def get_path(self, key: str) -> str:
        return self.config.get("Paths", key)
    
    def get_sys_config(self, key: str) -> str:
        return self.config.get("Configuracion", key)

    def get_video_float(self, key: str) -> float:
        return self.config.getfloat("Video", key)
        
    def get_video_int(self, key: str) -> int:
        return self.config.getint("Video", key)
    
    def get_all_config_as_dict(self) -> dict:
        """
        Convierte toda la configuración actual cargada en memoria 
        a un diccionario estándar de Python.
        """
        config_dict = {}
        for section in self.config.sections():
            # convertir cada sección en un sub-diccionario
            config_dict[section] = dict(self.config.items(section))
        return config_dict

    def export_config(self, output_path: str):
        """
        Guarda una copia exacta de la configuración usada en esta ejecución.
        """
        config_data = self.get_all_config_as_dict()
        save_json(config_data, output_path)
        print(f" [SISTEMA] Configuración guardado en: {output_path}")