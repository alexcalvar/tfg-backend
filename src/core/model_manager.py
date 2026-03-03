import os

from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from core.adapters.message_builders import CloudMessageBuilder,LocalMessageBuilder


class ModelManager:
    def __init__(self, vlm_provider, vlm_model_name, llm_model_name = None):
        self.vlm_name= vlm_model_name
        self.llm_name= llm_model_name
        self.vlm_provider_name = vlm_provider

    def load_vlm(self):
        match self.vlm_provider_name:
            case "google":

                api_key = os.getenv("GOOGLE_API_KEY")

                if not api_key:
                 raise ValueError(" Falta GOOGLE_API_KEY en .env")
                
                print(f" Conectando con Google en remoto: {self.vlm_name}")
            
                message_strategy = CloudMessageBuilder()
                
                modelo =  ChatGoogleGenerativeAI(
                    model=self.vlm_name, 
                    google_api_key=api_key,
                    temperature=0,
                    )
                
                return modelo, message_strategy
            
            case "ollama":
                
                print(f" Conectando con Ollama Local: {self.vlm_name}")

                message_strategy = LocalMessageBuilder()
                modelo = ChatOllama(
                    model=self.vlm_name,
                    temperature=0,
                    format="json"
                )

                return modelo, message_strategy
            
            case "openroute":
                print(f"Conectando con Open Route en remoto:  {self.vlm_name}")

                api_key_or=os.getenv("OPEN_ROUTE_API_KEY")

                if not api_key_or:
                 raise ValueError(" Falta OPEN_ROUTE_API_KEY en .env")

                message_strategy = CloudMessageBuilder()
                
                modelo =  ChatOpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    model=self.vlm_name,
                    temperature=0,
                    api_key=api_key_or
                )

                return modelo, message_strategy
            
            case _:
                raise ValueError(f" El proveedor VLM '{self.vlm_provider_name}' no está soportado en este Manager.")
            


    def load_llm(self):
        match self.llm_name:
            case "llama3-70b-8192" | "llama3-8b-8192" | "mixtral-8x7b-32768":

                return ChatGroq(
                    api_key=os.environ.get("GROQ_API_KEY"),
                    model=self.llm_name,
                    temperature=0
                )
            
            case _:
                raise ValueError(f" El modelo VLM '{self.llm_name}' no está soportado en este Manager.")