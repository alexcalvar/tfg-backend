import base64 
import json
import os


from langchain_core.language_models.chat_models import BaseChatModel

from src.core.message_strategies.message_builders import MessageBuilderStrategy

class VLMProcessor:
    def __init__(self, vlm_model_instance : BaseChatModel, selected_message_strategy : MessageBuilderStrategy, system_prompt):
        self.vlm = vlm_model_instance
        self.message_strategy = selected_message_strategy
        self.sys_prompt = system_prompt


    def analyze_frame(self, model_request : list[dict]) -> str:
        """recibe un layout genérico desde la estregia de procesamiento y lo pasa al messagebuilder."""
        messages = self.message_strategy.build_messages(self.sys_prompt, model_request)
        
        response = self.vlm.invoke(messages)

        return response.content
        
