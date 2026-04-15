import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

from src.data.enums import StrategyType
from src.core.pipeline import VLMPipeline 
from src.core.factories.model_factory import ModelFactory
from src.core.factories.processing_factory import ProcessingFactory
from src.postprocessing.postprocessing_algorithms.sliding_window import SlidingWindowNormalizer
from src.postprocessing.resums_logic.semantic_processor import SemanticAnalyzer
from src.utils.file_utils import load_json

class CLIModelTester:
    """Clase para ejecutar y probar modelos VLM en crudo por terminal, sin pasar por la API."""
    
    def __init__(self):
        load_dotenv()
        self.rutas_modelos = Path("configs/models_config.json")
        
        if not self.rutas_modelos.exists():
            print(f" [ERROR FATAL] No se encuentra el archivo de configuración en {self.rutas_modelos}")
            self.config_modelos = {"vlms": {}}
        else:
            self.config_modelos = load_json(str(self.rutas_modelos))




    # ==========================================
    # MÉTODOS PRIVADOS DE MENÚ (CLI)
    # ==========================================

    def _seleccionar_modelo(self) -> tuple[str, str]:
        """Muestra el menú de modelos y devuelve el proveedor y nombre seleccionados."""
        print("\n--- PASO 1: SELECCIÓN DE MODELO VLM ---")
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
            raise ValueError("No hay modelos configurados en tu archivo JSON.")

        for i, opcion in enumerate(opciones_menu, start=1):
            prov = opcion["proveedor"].upper()
            nombre = opcion["id_modelo"]
            desc = opcion["datos"].get("descripcion", opcion["datos"].get("model_string", ""))
            print(f"  {i} - [{prov}] {nombre}: {desc}")
            
        n = input("\n  Introduzca el número del modelo que desea usar: ")
        
        try:
            indice = int(n) - 1
            if indice < 0 or indice >= len(opciones_menu):
                raise IndexError()
            
            seleccion = opciones_menu[indice]
            print(f" [INFO] Modelo seleccionado: {seleccion['id_modelo']} vía {seleccion['proveedor']}")
            return seleccion["proveedor"], seleccion["id_modelo"]
            
        except (ValueError, IndexError):
            raise ValueError("Selección de modelo no válida.")



    def _seleccionar_estrategia(self) -> str:
        """Muestra el menú de estrategias y devuelve el valor del Enum seleccionado."""
        print("\n--- PASO 2: SELECCIÓN DE ESTRATEGIA DE PROCESAMIENTO ---")
        
        # Extraemos las opciones dinámicamente del Enum
        opciones_estrategia = list(StrategyType)
        
        for i, estrategia in enumerate(opciones_estrategia, start=1):
            print(f"  {i} - {estrategia.name} ({estrategia.value})")
            
        n = input("\n  Introduzca el número de la estrategia que desea usar: ")
        
        try:
            indice = int(n) - 1
            if indice < 0 or indice >= len(opciones_estrategia):
                raise IndexError()
            
            seleccion = opciones_estrategia[indice]
            print(f" [INFO] Estrategia seleccionada: {seleccion.name}")
            return seleccion.value
            
        except (ValueError, IndexError):
            raise ValueError("Selección de estrategia no válida.")




    # ==========================================
    # FLUJO PRINCIPAL DE EJECUCIÓN
    # ==========================================

    async def ejecutar_prueba(self, video_name: str, user_prompt: str):
        """Ejecuta el pipeline interactivo por terminal."""
        print("\n" + "="*50)
        print(" INICIANDO MODO DE PRUEBA LOCAL (CLI TFG)")
        print("="*50)

        ruta_video = os.path.join("datasets", "videos_test", video_name)
        if not os.path.exists(ruta_video):
            print(f"  [ERROR] No encuentro el vídeo en: {ruta_video}")
            return

        try:
            #  Menús interactivos
            vlm_provider, vlm_model_name = self._seleccionar_modelo()
            selected_process_stry = self._seleccionar_estrategia()

            llm_provider = "ollama"
            llm_model_name = "qwen2.5"
            
            #simula la decision del usuario de si quiere o no aplicar el algoritmo
            apply_alg = False
            
            # Ensamblaje de dependencias
            print("\n  Arrancando motores de IA...")
            
            #acordarse de cambiar por el factory
            algoritmo_normalizacion = SlidingWindowNormalizer(apply_alg)
            
            vlm_model, msg_strategy = ModelFactory().load_vlm(vlm_provider, vlm_model_name)
            llm_model = ModelFactory().load_llm(llm_provider, llm_model_name)
            process_strategy = ProcessingFactory().create_strategy(selected_process_stry)

            semantic_resum = SemanticAnalyzer(llm_instance=llm_model, user_prompt= user_prompt)
            

            pipeline = VLMPipeline(
                model_instance=vlm_model, 
                provider_name=vlm_provider, 
                message_strategy=msg_strategy, 
                processing_strategy=process_strategy, 
                postprocessing_strategy=semantic_resum
            ) 
            
        
            #  Ejecución del análisis
            await pipeline.process_video(ruta_video, user_prompt)
            print("\n  Prueba finalizada con éxito. Revisa la carpeta de proyectos.")
        
        except ValueError as ve:
            print(f"  [CANCELADO] {ve} Saliendo del sistema...")
        except Exception as e:
            print(f"  [CRÍTICO] Error en la ejecución del pipeline: {e}")
            import traceback
            traceback.print_exc()



if __name__ == "__main__":
    tester = CLIModelTester()
    
    # Parámetros definidos en la fase de análisis 
    VIDEO_DE_PRUEBA = "video_lobos.mp4" 
    PROMPT_DE_PRUEBA = "Resume los contenidos y destaca los animales que veas"
    
    asyncio.run(tester.ejecutar_prueba(VIDEO_DE_PRUEBA, PROMPT_DE_PRUEBA))