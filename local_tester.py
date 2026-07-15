import os
import time
import asyncio
import signal
from pathlib import Path
from dotenv import load_dotenv

from src.data.enums import StrategyType
from src.core.pipeline import VLMPipeline 
from src.core.factories.model_factory import ModelFactory
from src.core.frame_providers.video_loader import VideoLoader
from src.core.factories.processing_factory import ProcessingFactory
from src.postprocessing.postprocessing_algorithms.sliding_window import SlidingWindowNormalizer
from src.postprocessing.postprocessing_algorithms.state_lock import StateLockNormalizer
from src.postprocessing.resums_logic.semantic_processor import SemanticAnalyzer
from src.utils.file_utils import load_json, ensure_dir
from src.utils.config_loader import ConfigLoader

from src.utils.logger import get_logger

logger = get_logger(__name__)

class CLIModelTester:
    """Clase para ejecutar y probar modelos VLM en crudo por terminal, sin pasar por la API."""
    
    def __init__(self):
        load_dotenv()
        self.config = ConfigLoader()
        self.rutas_modelos = Path("configs/models_config.json")
        
        if not self.rutas_modelos.exists():
            logger.critical(f"No se encuentra el archivo de configuración en {self.rutas_modelos}")
            self.config_modelos = {"vlms": {}}
        else:
            self.config_modelos = load_json(str(self.rutas_modelos))

    def _seleccionar_video(self) -> str:
        """Muestra el menú de vídeos disponibles en datasets/videos_test y devuelve la ruta elegida."""
        print("\n--- PASO 0: SELECCIÓN DE VÍDEO ---")
        carpeta_videos = Path("datasets") / "videos_test"

        if not carpeta_videos.exists():
            raise ValueError(f"No se encuentra la carpeta de vídeos de prueba: {carpeta_videos}")

        extensiones_validas = {".mp4", ".avi", ".mov", ".mkv"}
        videos_disponibles = sorted(
            f for f in carpeta_videos.iterdir() if f.suffix.lower() in extensiones_validas
        )

        if not videos_disponibles:
            raise ValueError(f"No hay vídeos disponibles en {carpeta_videos}")

        for i, video in enumerate(videos_disponibles, start=1):
            print(f"  {i} - {video.name}")

        n = input("\n  Introduzca el número del vídeo que desea analizar: ")

        try:
            indice = int(n) - 1
            if indice < 0 or indice >= len(videos_disponibles):
                raise IndexError()

            seleccion = videos_disponibles[indice]
            logger.info(f"Vídeo seleccionado: {seleccion.name}")
            return str(seleccion)

        except (ValueError, IndexError):
            raise ValueError("Selección de vídeo no válida.")

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
            logger.info(f"Modelo seleccionado: {seleccion['id_modelo']} vía {seleccion['proveedor']}")
            return seleccion["proveedor"], seleccion["id_modelo"]
            
        except (ValueError, IndexError):
            raise ValueError("Selección de modelo no válida.")

    def _seleccionar_modelo_llm(self) -> tuple[str, str]:
        """Muestra el menú de modelos LLM (resumen semántico) y devuelve el proveedor y nombre seleccionados."""
        print("\n--- SELECCIÓN DE MODELO LLM (RESUMEN SEMÁNTICO) ---")
        opciones_menu = []
        llms_config = self.config_modelos.get("llms", {})

        for proveedor, modelos in llms_config.items():
            for clave_modelo, datos in modelos.items():
                opciones_menu.append({
                    "proveedor": proveedor,
                    "id_modelo": clave_modelo,
                    "datos": datos
                })

        if not opciones_menu:
            raise ValueError("No hay modelos LLM configurados en tu archivo JSON.")

        for i, opcion in enumerate(opciones_menu, start=1):
            prov = opcion["proveedor"].upper()
            nombre = opcion["id_modelo"]
            desc = opcion["datos"].get("descripcion", opcion["datos"].get("model_string", ""))
            print(f"  {i} - [{prov}] {nombre}: {desc}")

        n = input("\n  Introduzca el número del modelo LLM que desea usar: ")

        try:
            indice = int(n) - 1
            if indice < 0 or indice >= len(opciones_menu):
                raise IndexError()

            seleccion = opciones_menu[indice]
            logger.info(f"Modelo LLM seleccionado: {seleccion['id_modelo']} vía {seleccion['proveedor']}")
            return seleccion["proveedor"], seleccion["id_modelo"]

        except (ValueError, IndexError):
            raise ValueError("Selección de modelo LLM no válida.")

    def _seleccionar_estrategia(self) -> str:
        """Muestra el menú de estrategias y devuelve el valor del Enum seleccionado."""
        print("\n--- PASO 2: SELECCIÓN DE ESTRATEGIA DE PROCESAMIENTO ---")
        
        opciones_estrategia = list(StrategyType)
        
        for i, estrategia in enumerate(opciones_estrategia, start=1):
            print(f"  {i} - {estrategia.name} ({estrategia.value})")
            
        n = input("\n  Introduzca el número de la estrategia que desea usar: ")
        
        try:
            indice = int(n) - 1
            if indice < 0 or indice >= len(opciones_estrategia):
                raise IndexError()
            
            seleccion = opciones_estrategia[indice]
            logger.info(f"Estrategia seleccionada: {seleccion.name}")
            return seleccion.value
            
        except (ValueError, IndexError):
            raise ValueError("Selección de estrategia no válida.")

    # ==========================================
    # FLUJO PRINCIPAL DE EJECUCIÓN
    # ==========================================

    async def ejecutar_prueba(self):
        """Ejecuta el pipeline interactivo por terminal, pidiendo todos los datos al usuario."""
        print("\n" + "="*50)
        print(" INICIANDO MODO DE PRUEBA LOCAL (CLI TFG)")
        print("="*50)

        try:
    
            ruta_video = self._seleccionar_video()
            
            vlm_provider, vlm_model_name = self._seleccionar_modelo()
            selected_process_stry = self._seleccionar_estrategia()

            user_prompt = input("\n  Introduzca la consulta/prompt que quiere aplicar al vídeo: ")
            
            run_id = f"cli_project_{int(time.time())}"
            base_run_dir = os.path.join(self.config.get_path("projects_folder"), run_id)
            frames_dir = os.path.join(base_run_dir, "frames")
            
            ensure_dir(base_run_dir)
            ensure_dir(frames_dir)

            logger.info("==========INICIANDO EJECUCIÓN CON CLI ================")
            logger.info("Arrancando motores de IA y ensamblando dependencias...")

            while True:
                interval_input = input("\n  Introduzca el intervalo de tiempo (en segundos) entre frames a analizar: ")
                try:
                    interval = float(interval_input)
                    break
                except ValueError:
                    print("  [ERROR] Introduce un número válido (ejemplo: 2.0).")

            frame_provider = VideoLoader(ruta_video, frames_dir, interval)
            
            vlm_model, msg_strategy = ModelFactory().load_vlm(vlm_provider, vlm_model_name)
            process_strategy = ProcessingFactory().create_strategy(selected_process_stry)

            print("\n--- PASO 3: SELECCIÓN DE POST-PROCESAMIENTO ---")
            print("  1 - DETECCION DE EVENTOS ")
            print("  2 - LOGICA DE RESUMENES ")
            selcted_postprocessing_strategy = input("\n  Seleccione la funcionalidad de postprocesado: ")

            match selcted_postprocessing_strategy:
                case "1":

                    print("\n--- PASO 4: SELECCIÓN DEL ALGORITMO ---")
                    print("  1 - VENTANA DESLIZANTE ")
                    print("  2 - ESTADOS")
                    print("  3 - Sin algoritmo")
                    selected_alg = input("\n  Seleccione el algoritmo: ")

                    match selected_alg:
                        case "1" : 
                            postprocess_strategy = SlidingWindowNormalizer(True, interval)
                        
                        case "2":
                            postprocess_strategy = StateLockNormalizer(True, interval)
                        
                        case "3":
                            postprocess_strategy = StateLockNormalizer(False, interval)

                        case _:
                            raise ValueError("Selección de algoritmo no válida.")

                case "2":
                    llm_provider, llm_model_name = self._seleccionar_modelo_llm()
                    llm_model = ModelFactory().load_llm(llm_provider, llm_model_name)
                    postprocess_strategy = SemanticAnalyzer(llm_instance=llm_model, user_prompt=user_prompt, interval_time=interval)
                case _:
                    raise ValueError("Opción de post-procesamiento no válida.")
            
            pipeline = VLMPipeline(
                model_instance=vlm_model, 
                provider_name=vlm_provider,
                frame_provider=frame_provider, 
                message_strategy=msg_strategy, 
                processing_strategy=process_strategy, 
                postprocessing_strategy=postprocess_strategy,
                base_run_dir=base_run_dir
            ) 
            
           
            cancel_event = asyncio.Event()
            
          
            loop = asyncio.get_running_loop()
            try:
                loop.add_signal_handler(
                    signal.SIGINT, 
                    lambda: (logger.warning("Cancelación detectada..."), cancel_event.set())
                )
                
            except NotImplementedError:
                
                logger.debug("La captura asíncrona de SIGINT no está disponible en este sistema operativo.")
        
            
            await pipeline.process_video(user_prompt, cancel_event)
            
            if cancel_event.is_set():
                logger.warning("El proceso se canceló cooperativamente antes de terminar.")
            else:
                logger.info("Prueba finalizada con éxito. Revisa la carpeta de proyectos.")
        
        except ValueError as ve:
            logger.warning(f"Ejecución cancelada por el usuario o validación: {ve}")
        except Exception as e:
            logger.exception(f"Error en la ejecución del pipeline: {e}")

if __name__ == "__main__":
    tester = CLIModelTester()

   
    asyncio.run(tester.ejecutar_prueba())