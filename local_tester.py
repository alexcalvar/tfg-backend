import os
import asyncio
from dotenv import load_dotenv

from src.core.pipeline import VLMPipeline 
from src.core.model_manager import ModelManager
from src.utils.file_utils import load_json

class CLIModelTester:
    """Clase para ejecutar y probar modelos VLM en crudo por terminal, sin pasar por la API."""
    
    def __init__(self):
        
        load_dotenv()
        
        # Cargar configuraciones en memoria 
        self.rutas_modelos = "configs/models_config.json"
        
        if not os.path.exists(self.rutas_modelos):
            print(f" [ERROR FATAL] No se encuentra el archivo de configuración en {self.rutas_modelos}")
            self.config_modelos = {"vlms": {}}
        else:
            self.config_modelos = load_json(self.rutas_modelos)

    async def ejecutar_prueba(self, video_name: str, user_prompt: str):
        """Ejecuta el pipeline interactivo por terminal."""
        print("\n" + "="*50)
        print(" INICIANDO MODO DE PRUEBA LOCAL (CLI TFG)")
        print("="*50)

        ruta_video = os.path.join("datasets", "videos_test", video_name)
        if not os.path.exists(ruta_video):
            print(f"  ERROR: No encuentro el vídeo en: {ruta_video}")
            print(" Revisa que el nombre sea correcto y esté en la carpeta datasets/videos_test.")
            return

        print("\n Seleccione el VLM que desea utilizar:")
        
        # Obtenemos la lista de todas los modelos del json
        modelos_disponibles = list(self.config_modelos.get("vlms", {}).keys())
        
        if not modelos_disponibles:
            print("  No hay modelos configurados en tu archivo JSON.")
            return

        for i, clave_modelo in enumerate(modelos_disponibles, start=1):
            desc = self.config_modelos["vlms"][clave_modelo].get("model_string", clave_modelo)
            print(f"  {i} - {desc}")
            
        n = input("\n  Introduzca el número del modelo que desea usar: ")
        
        try:
            indice = int(n) - 1
            if indice < 0 or indice >= len(modelos_disponibles):
                raise ValueError()
                
            clave_seleccionada = modelos_disponibles[indice]
            datos_modelo = self.config_modelos["vlms"][clave_seleccionada]
            
            vlm_model_name = datos_modelo["model_string"]
            vlm_provider = datos_modelo["provider"]
            print(f"\n [INFO] Configuración cargada: {datos_modelo.get('descripcion', vlm_model_name)}")
            
        except (ValueError, IndexError):
            print("  [ERROR] Selección no válida. Saliendo del sistema...")
            return

        print("\n  Arrancando motores de IA...")
        try:
            vlm_model, strategy = ModelManager(vlm_provider, vlm_model_name).load_vlm()

            pipeline = VLMPipeline(vlm_model, vlm_provider, strategy) 
            
            # Ejecutamos el pipeline de forma 
            await pipeline.process_video(ruta_video, user_prompt)
            print("\n  Prueba finalizada con éxito. Revisa la carpeta de proyectos.")
        
        except Exception as e:
            print(f"  Ocurrió un error inesperado en el pipeline: {e}")

# ==========================================
# PUNTO DE ENTRADA (CLI)
# ==========================================
if __name__ == "__main__":
    tester = CLIModelTester()
    
    # Parámetros duros para la prueba
    VIDEO_DE_PRUEBA = "video_perro_prueba.mp4" 
    PROMPT_DE_PRUEBA = "Dime si ves un perro en la imagen"
    
    # Arrancamos el bucle asíncrono
    asyncio.run(tester.ejecutar_prueba(VIDEO_DE_PRUEBA, PROMPT_DE_PRUEBA))