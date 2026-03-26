from langchain_core.language_models.chat_models import BaseChatModel
from src.core.message_strategies.message_builders import MessageStrategy

class VLMProcessor:
    def __init__(self, vlm_model: BaseChatModel, message_strategy: MessageStrategy, system_prompt: str):
        self.vlm = vlm_model
        self.message_strategy = message_strategy
        self.system_prompt = system_prompt



    def process_layout(self, layout: list[dict]) -> str:
        """
        Recibe un layout genérico desde la estrategia de procesamiento, construye la petición en el formato especifico
         del tipo de procesamiento y devuelve la respuesta del VLM en crudo
        """
        # generar el mensaje con el formato especifico
        messages = self.message_strategy.build_messages(self.system_prompt, layout)
        
        # ejecutar el modelo 
        response = self.vlm.invoke(messages)

        # retornar el contenido de la respuesta del modelo
        return response.content