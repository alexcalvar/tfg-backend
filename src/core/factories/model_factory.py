import os

from langchain_community.chat_models import ChatLlamaCpp
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

from llama_cpp import Llama

from src.core.model_adapters.llamacpp_adapter import CustomVisionLlamaCpp
from src.core.message_strategies.message_builders import CloudMessageBuilder,LocalMessageBuilder, MessageStrategy
from src.core.model_adapters.vlm_handler import VLM_HANDLERS

from src.data.enums import LLMProvider, VLMProvider

from src.utils.config_loader import ConfigLoader
from src.utils.file_utils import load_json 

from src.utils.logger import get_logger

logger = get_logger(__name__)

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
            logger.critical(f"El proveedor VLM '{vlm_provider_str}' no está soportado.")
            raise ValueError(f"El proveedor VLM '{vlm_provider_str}' no está soportado.")

        # configuracion del modelo especifico
        try:
            vlm_model_config = self.models_config["vlms"][provider.value][vlm_name]
        except KeyError:
            logger.exception(f"No se encontró configuración para el modelo '{vlm_name}' en el proveedor '{provider.value}'.")
            raise ValueError(f"No se encontró configuración para el modelo '{vlm_name}' en el proveedor '{provider.value}'.")

        match provider:
            case VLMProvider.LLAMACPP:
                modelo = self._build_llamacpp(vlm_model_config)
                message_strategy = LocalMessageBuilder()
                return modelo, message_strategy
                
            case VLMProvider.GOOGLE:
                modelo = self._build_google(vlm_model_config, vlm_name)
                message_strategy = CloudMessageBuilder()
                return modelo, message_strategy
                
            case VLMProvider.OLLAMA:
                modelo =  self._build_ollama(vlm_model_config, vlm_name)
                message_strategy = LocalMessageBuilder()
                return modelo, message_strategy
                
            case VLMProvider.OPENROUTE:
                modelo = self._build_openroute(vlm_model_config, vlm_name)
                message_strategy = CloudMessageBuilder()
                return modelo, message_strategy




    def load_llm(self, llm_provider_name : str, llm_name : str) -> BaseChatModel:
        # string a enum
        try:
            provider = LLMProvider(llm_provider_name.lower())
        except ValueError:
            logger.critical(f"El proveedor LLM '{llm_provider_name}' no está soportado.")
            raise ValueError(f"El proveedor LLM '{llm_provider_name}' no está soportado.")

        # configuracion del modelo especifico
        try:
            llm_model_config = self.models_config["llms"][provider.value][llm_name]
        except KeyError:
            logger.exception(f"No se encontró configuración para el modelo '{llm_name}' en el proveedor '{provider.value}'.")
            raise ValueError(f"No se encontró configuración para el modelo '{llm_name}' en el proveedor '{provider.value}'.")

        match provider:
            case LLMProvider.LLAMACPP:
                return self._build_llamacpp_llm(llm_model_config)
                
            case LLMProvider.GOOGLE:
                return self._build_google(llm_model_config, llm_name)
                
            case LLMProvider.OLLAMA:
                return self._build_ollama(llm_model_config, llm_name)
                
            case LLMProvider.OPENROUTE:
                return self._build_openroute(llm_model_config, llm_name)
            
            case LLMProvider.GROQ:
                return self._build_groq(llm_model_config, llm_name)
                




# ----- METODOS DE CREACION DEL MODELOS -------
# para llamacpp se diferencia de si es para llm o vlm por la necesisdad de añadir el complemento 
#visual para el vlm

    def _build_llamacpp(self, config: dict) -> BaseChatModel:
        logger.info("Conectando con Llama.cpp local para VLM ...")
        
        handler_type = config["handler_type"]
        handler = VLM_HANDLERS.get(handler_type)

        if not handler:
            logger.critical(f"Arquitectura: El handler '{handler_type}' no está registrado en VLM_HANDLERS.")
            raise ValueError(f"Arquitectura: El handler '{handler_type}' no está registrado en VLM_HANDLERS.")

        visual_adapter_path = config["vision_projector_path"]
        visual_adapter = handler(clip_model_path=visual_adapter_path)


        modelo_llamacpp = Llama(
            model_path=config["model_path"],
            chat_handler=visual_adapter,
            n_ctx=config["n_ctx"],
            verbose=False
        )

        adapt_langchain = CustomVisionLlamaCpp(cliente_nativo=modelo_llamacpp)
        return adapt_langchain 
    


    def _build_llamacpp_llm(self, config: dict) -> BaseChatModel:
        
        logger.info("Conectando con Llama.cpp local para LLM ...")
        
        return ChatLlamaCpp(
            model_path=config["model_path"],
            n_ctx=config["n_ctx"],
            temperature=0
        )




    def _build_google(self, config: dict, name: str) -> BaseChatModel:
        logger.info(f"Conectando con Google en remoto: {name}")
        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            logger.critical("Falta GOOGLE_API_KEY en el archivo .env")
            raise ValueError("Falta GOOGLE_API_KEY en el archivo .env")
        
        modelo = ChatGoogleGenerativeAI(
            model=config["model_string"], 
            google_api_key=api_key,
            temperature=0,
        )
        
        return modelo 




    def _build_ollama(self, config: dict, name: str) -> BaseChatModel:
        logger.info(f"Conectando con Ollama Local: {name}")
        
        modelo = ChatOllama(
            model=config["model_string"],
            temperature=0,
        )
        return modelo 




    def _build_openroute(self, config: dict, name: str) -> BaseChatModel:
        logger.info(f"Conectando con OpenRouter en remoto: {name}")
        api_key_or = os.getenv("OPEN_ROUTE_API_KEY")

        if not api_key_or:
            logger.critical("Falta OPEN_ROUTE_API_KEY en el archivo .env")
            raise ValueError("Falta OPEN_ROUTE_API_KEY en el archivo .env")

        modelo = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            model=config["model_string"],
            temperature=0,
            api_key=api_key_or
        )
        return modelo 



    def _build_groq(self, config: dict, name: str) -> BaseChatModel:
        logger.info(f"Conectando con Groq en remoto: {name}")
        api_key_groq = os.getenv("GROQ_API_KEY")
        
        if not api_key_groq:
            logger.critical("Falta GROQ_API_KEY en el archivo .env")
            raise ValueError("Falta GROQ_API_KEY en el archivo .env")

        return ChatGroq(
            model=config["model_string"],
            temperature=0,
            api_key=api_key_groq
        )