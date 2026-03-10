import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

from src.core.pipeline import VLMPipeline 
from src.core.model_manager import ModelManager
from src.utils.file_utils import load_json

class CLIModelTester:
    """Clase para ejecutar y probar modelos VLM en crudo por terminal, sin pasar por la API."""
    
    def __init__(self):
        load_dotenv()
        # Usamos Path para mayor robustez en las rutas 
        self.rutas_modelos = Path("configs/models_config.json")
        
        if not self.rutas_modelos.exists():
            print(f" [ERROR FATAL] No se encuentra el archivo de configuración en {self.rutas_modelos}")
            self.config_modelos = {"vlms": {}}
        else:
            self.config_modelos = load_json(str(self.rutas_modelos))

    async def ejecutar_prueba(self, video_name: str, user_prompt: str):
        """Ejecuta el pipeline interactivo por terminal."""
        print("\n" + "="*50)
        print(" INICIANDO MODO DE PRUEBA LOCAL (CLI TFG)")
        print("="*50)

        ruta_video = os.path.join("datasets", "videos_test", video_name)
        if not os.path.exists(ruta_video):
            print(f"  ERROR: No encuentro el vídeo en: {ruta_video}")
            return

        print("\n Modelos VLM disponibles en la configuración:")
        
        # 1. Aplanamos la jerarquía del JSON para que sea seleccionable
        opciones_menu = []
        vlms_config = self.config_modelos.get("vlms", {})
        
        for proveedor, modelos in vlms_config.items():
            for clave_modelo, datos in modelos.items():
                opciones_menu.append({
                    "proveedor": proveedor,
                    "id_modelo": clave_modelo,
                    "datos": datos
                })

        if not opciones_menu:
            print("  [ERROR] No hay modelos configurados en tu archivo JSON.")
            return

        # 2. Mostramos el menú al usuario
        for i, opcion in enumerate(opciones_menu, start=1):
            prov = opcion["proveedor"].upper()
            nombre = opcion["id_modelo"]
            desc = opcion["datos"].get("descripcion", opcion["datos"].get("model_string", ""))
            print(f"  {i} - [{prov}] {nombre}: {desc}")
            
        n = input("\n  Introduzca el número del modelo que desea usar: ")
        
        try:
            indice = int(n) - 1
            if indice < 0 or indice >= len(opciones_menu):
                raise ValueError()
                
            seleccion = opciones_menu[indice]
            vlm_provider = seleccion["proveedor"]
            vlm_model_name = seleccion["id_modelo"]
            
            print(f"\n [INFO] Seleccionado: {vlm_model_name} vía {vlm_provider}")
            
        except (ValueError, IndexError):
            print("  [ERROR] Selección no válida. Saliendo del sistema...")
            return

        print("\n  Arrancando motores de IA...")
        try:
            # 3. Inyectamos las dependencias en el ModelManager [cite: 1, 2]
            # Pasamos proveedor y el nombre de la clave para que el Factory busque en el JSON
            manager = ModelManager(vlm_provider, vlm_model_name)
            vlm_model, strategy = manager.load_vlm()

            # 4. Orquestación del Pipeline [cite: 223, 406]
            pipeline = VLMPipeline(vlm_model, vlm_provider, strategy) 
            
            # El procesamiento asíncrono es un requisito no funcional (RNF-04) [cite: 508]
            await pipeline.process_video(ruta_video, user_prompt)
            print("\n  Prueba finalizada con éxito. Revisa la carpeta de proyectos.")
        
        except Exception as e:
            # Captura de errores para evitar cierres inesperados (RF-11) [cite: 501, 503]
            print(f"  [CRÍTICO] Error en la ejecución del pipeline: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    tester = CLIModelTester()
    
    # Parámetros definidos en la fase de análisis (Sprint 1) [cite: 300, 311]
    VIDEO_DE_PRUEBA = "video_perro_prueba.mp4" 
    PROMPT_DE_PRUEBA = "Dime si ves un perro en la imagen"
    
    asyncio.run(tester.ejecutar_prueba(VIDEO_DE_PRUEBA, PROMPT_DE_PRUEBA))