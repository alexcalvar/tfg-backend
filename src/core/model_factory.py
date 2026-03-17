import os

from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from llama_cpp import Llama

from src.core.model_adapters.llamacpp_adapter import CustomVisionLlamaCpp
from src.core.message_strategies.message_builders import CloudMessageBuilder,LocalMessageBuilder
from src.core.model_adapters.vlm_handler import VLM_HANDLERS
from src.utils.config_loader import ConfigLoader

from src.utils.file_utils import load_json 


class ModelFactory:
    def __init__(self):
        self.config = ConfigLoader()
        config_folder_path = self.config.get_path("config_folder")
        models_config_path = os.path.join(config_folder_path, "models_config.json")
        self.models_config = load_json(models_config_path)

    def load_vlm(self,vlm_provider_name, vlm_name):

        vlms_configs = self.models_config["vlms"]

        match vlm_provider_name:
            
            case "llamacpp":
                
                vlm_model = vlms_configs[vlm_provider_name][vlm_name]

                print("Conectando con Llama cpp local:")

                #extraer handler especifico del modelo desde la configuracion de modelos
                handler_type = vlm_model["handler_type"]
                handler = VLM_HANDLERS.get(handler_type)

                if not handler:
                    raise ValueError(f"Arquitectura: El handler '{handler_type}' no está registrado en VLM_HANDLERS.")

                visual_adapter_path = vlm_model["vision_projector_path"]
                visual_adapter = handler(clip_model_path=visual_adapter_path)

                message_strategy = LocalMessageBuilder()

                modelo_llamacpp = Llama(
                    model_path=vlm_model["model_path"],
                    chat_handler=visual_adapter,
                    n_ctx=vlm_model["n_ctx"],
                    verbose=False
                )

                adapt_langchain = CustomVisionLlamaCpp(cliente_nativo=modelo_llamacpp)

                return adapt_langchain, message_strategy


            case "google":
                
                vlm_model = vlms_configs[vlm_provider_name][vlm_name]

                api_key = os.getenv("GOOGLE_API_KEY")

                if not api_key:
                 raise ValueError(" Falta GOOGLE_API_KEY en .env")
                
                print(f" Conectando con Google en remoto: {vlm_name}")
            
                message_strategy = CloudMessageBuilder()
                
                modelo =  ChatGoogleGenerativeAI(
                    model=vlm_model["model_string"], 
                    google_api_key=api_key,
                    temperature=0,
                    )
                
                return modelo, message_strategy
            
            case "ollama":

                vlm_model = vlms_configs[vlm_provider_name][vlm_name]
                
                print(f" Conectando con Ollama Local: {vlm_name}")

                message_strategy = LocalMessageBuilder()
                modelo = ChatOllama(
                    model=vlm_model["model_string"],
                    temperature=0,
                    #format="json"
                )

                return modelo, message_strategy
            
            case "openroute":

                vlm_model = vlms_configs[vlm_provider_name][vlm_name]

                print(f"Conectando con Open Route en remoto:  {vlm_name}")

                api_key_or=os.getenv("OPEN_ROUTE_API_KEY")

                if not api_key_or:
                 raise ValueError(" Falta OPEN_ROUTE_API_KEY en .env")

                message_strategy = CloudMessageBuilder()
                
                modelo =  ChatOpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    model=vlm_model["model_string"],
                    temperature=0,
                    api_key=api_key_or
                )

                return modelo, message_strategy
            
            case _:
                raise ValueError(f" El proveedor VLM '{vlm_provider_name}' no está soportado en este Manager.")
            


    def load_llm(self, llm_provider_name, llm_name):
        match llm_provider_name:
            case "llama3-70b-8192" | "llama3-8b-8192" | "mixtral-8x7b-32768":

                return ChatGroq(
                    api_key=os.environ.get("GROQ_API_KEY"),
                    model=llm_name,
                    temperature=0
                )
            
            case _:
                raise ValueError(f" El modelo VLM '{llm_name}' no está soportado en este Manager.")