import os

from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

from llama_cpp import Llama

from src.core.model_adapters.llamacpp_adapter import CustomVisionLlamaCpp
from src.core.message_strategies.message_builders import CloudMessageBuilder,LocalMessageBuilder, MessageStrategy
from src.core.model_adapters.vlm_handler import VLM_HANDLERS

from src.data.enums import VLMProvider

from src.utils.config_loader import ConfigLoader
from src.utils.file_utils import load_json 


class ModelFactory:
    def __init__(self):
        self.config = ConfigLoader()
        config_folder_path = self.config.get_path("config_folder")
        models_config_path = os.path.join(config_folder_path, "models_config.json")
        self.models_config = load_json(models_config_path)

    def load_vlm(self, vlm_provider_str: str, vlm_name: str) -> tuple[BaseChatModel,MessageStrategy]:
        """Punto de entrada para cargar modelos VLM"""
        # string a enum
        try:
            provider = VLMProvider(vlm_provider_str.lower())
        except ValueError:
            raise ValueError(f"El proveedor VLM '{vlm_provider_str}' no está soportado.")

        # configuracion del modelo especifico
        try:
            vlm_model_config = self.models_config["vlms"][provider.value][vlm_name]
        except KeyError:
            raise ValueError(f"No se encontró configuración para el modelo '{vlm_name}' en el proveedor '{provider.value}'.")

        match provider:
            case VLMProvider.LLAMACPP:
                return self._build_llamacpp(vlm_model_config)
                
            case VLMProvider.GOOGLE:
                return self._build_google(vlm_model_config, vlm_name)
                
            case VLMProvider.OLLAMA:
                return self._build_ollama(vlm_model_config, vlm_name)
                
            case VLMProvider.OPENROUTE:
                return self._build_openroute(vlm_model_config, vlm_name)




    def load_llm(self, llm_provider_name : str, llm_name : str):
        pass




# ----- METODOS DE CREACION DEL MODELOS VLM  Y SU ESTRATEGUA DE MSG -------

    def _build_llamacpp(self, config: dict) -> tuple[BaseChatModel,MessageStrategy]:
        print("[INFO] Conectando con Llama.cpp local...")
        
        handler_type = config["handler_type"]
        handler = VLM_HANDLERS.get(handler_type)

        if not handler:
            raise ValueError(f"Arquitectura: El handler '{handler_type}' no está registrado en VLM_HANDLERS.")

        visual_adapter_path = config["vision_projector_path"]
        visual_adapter = handler(clip_model_path=visual_adapter_path)

        message_strategy = LocalMessageBuilder()

        modelo_llamacpp = Llama(
            model_path=config["model_path"],
            chat_handler=visual_adapter,
            n_ctx=config["n_ctx"],
            verbose=False
        )

        adapt_langchain = CustomVisionLlamaCpp(cliente_nativo=modelo_llamacpp)
        return adapt_langchain, message_strategy




    def _build_google(self, config: dict, name: str) -> tuple[BaseChatModel,MessageStrategy]:
        print(f" [INFO] Conectando con Google en remoto: {name}")
        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError("Falta GOOGLE_API_KEY en el archivo .env")
        
        message_strategy = CloudMessageBuilder()
        modelo = ChatGoogleGenerativeAI(
            model=config["model_string"], 
            google_api_key=api_key,
            temperature=0,
        )
        return modelo, message_strategy




    def _build_ollama(self, config: dict, name: str) -> tuple[BaseChatModel,MessageStrategy]:
        print(f" [INFO] Conectando con Ollama Local: {name}")
        message_strategy = LocalMessageBuilder()
        
        modelo = ChatOllama(
            model=config["model_string"],
            temperature=0,
        )
        return modelo, message_strategy




    def _build_openroute(self, config: dict, name: str) -> tuple[BaseChatModel,MessageStrategy]:
        print(f" [INFO] Conectando con OpenRouter en remoto: {name}")
        api_key_or = os.getenv("OPEN_ROUTE_API_KEY")

        if not api_key_or:
            raise ValueError("Falta OPEN_ROUTE_API_KEY en el archivo .env")

        message_strategy = CloudMessageBuilder()
        modelo = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            model=config["model_string"],
            temperature=0,
            api_key=api_key_or
        )
        return modelo, message_strategy



