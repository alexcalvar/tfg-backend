from abc import ABC, abstractmethod
from langchain_core.messages import HumanMessage, SystemMessage

from src.data.validators import VideoFrame

class MessageBuilderStrategy(ABC):
    """
    Clase abstracta que define el contrato para todos los constructores de mensajes.
    """
    
    @abstractmethod
    def build_messages(self, system_prompt: str, user_prompt: str, images_b64: list[VideoFrame]) -> list:
        pass

    def _format_base64_url(self, image_b64: str, mime_type: str = "image/jpeg") -> str:
        return f"data:{mime_type};base64,{image_b64}"


class CloudMessageBuilder(MessageBuilderStrategy):
    """
    Estrategia para modelos en la nube.
    Estos modelos  entienden la separación entre SystemMessage y HumanMessage, así como el formato de diccionario estándar.
    """
    def build_messages(self, system_prompt: str, user_prompt: str, images_b64: list[VideoFrame]) -> list:

        human_content = [{"type": "text", "text": user_prompt}]

        for frame in images_b64:
            human_content.append({"type": "text", "text": f"Frame {frame.frame_id}:"})
            human_content.append({
                "type": "image_url", 
                "image_url": {"url": self._format_base64_url(frame.img_b64)}
            })
            
        
        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content)
        ]


class LocalMessageBuilder(MessageBuilderStrategy):
    """
    Estrategia para modelos locales .
    Soluciona la "fuga de abstracción": a menudo ignoran el SystemMessage o 
    alucinan si hay múltiples roles. Se unifican las instrucciones y la 
    pregunta en un único HumanMessage para mayor estabilidad.
    """
    def build_messages(self, system_prompt: str, user_prompt: str, images_b64: list[VideoFrame]) -> list:

        combined_prompt = f"{system_prompt}\n\nInstrucción del usuario: {user_prompt}"
        
        human_content = [{"type": "text", "text": combined_prompt}]
        
        for frame in images_b64:
            human_content.append({"type": "text", "text": f"Frame {frame.frame_id}:"})
            human_content.append({
                "type": "image_url", 
                "image_url": {"url": self._format_base64_url(frame.img_b64)}
            })
        
        return [HumanMessage(content=human_content)]
        