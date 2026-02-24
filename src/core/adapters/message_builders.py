from abc import ABC, abstractmethod
from langchain_core.messages import HumanMessage, SystemMessage

class MessageBuilderStrategy(ABC):
    """
    Clase abstracta que define el contrato (interfaz) para todos los constructores de mensajes.
    Cualquier estrategia nueva debe implementar el método build_messages.
    """
    
    @abstractmethod
    def build_messages(self, system_prompt: str, user_prompt: str, image_b64: str) -> list:
        pass

    def _format_base64_url(self, image_b64: str, mime_type: str = "image/jpeg") -> str:
        return f"data:{mime_type};base64,{image_b64}"


class CloudMessageBuilder(MessageBuilderStrategy):
    """
    Estrategia para modelos en la nube (Google Gemini, OpenRouter, OpenAI).
    Estos modelos son grandes, robustos y entienden perfectamente la separación
    entre SystemMessage y HumanMessage, así como el formato de diccionario estándar.
    """
    def build_messages(self, system_prompt: str, user_prompt: str, image_b64: str) -> list:
        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": self._format_base64_url(image_b64)}}
            ])
        ]


class LocalMessageBuilder(MessageBuilderStrategy):
    """
    Estrategia para modelos locales pequeños (Llava, Moondream vía Ollama).
    Soluciona la "fuga de abstracción": a menudo ignoran el SystemMessage o 
    alucinan si hay múltiples roles. Aquí colapsamos las instrucciones y la 
    pregunta en un único HumanMessage para mayor estabilidad.
    """
    def build_messages(self, system_prompt: str, user_prompt: str, image_b64: str) -> list:
        
        combined_prompt = f"{system_prompt}\n\nInstrucción del usuario: {user_prompt}"
        
        return [
            HumanMessage(content=[
                {"type": "text", "text": combined_prompt},
                {"type": "image_url", "image_url": {"url": self._format_base64_url(image_b64)}}
            ])
        ]