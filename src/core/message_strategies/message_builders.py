from abc import ABC, abstractmethod
from langchain_core.messages import HumanMessage, SystemMessage

from src.data.validators import FramesPath
from src.utils.file_utils import encode_image_base64

class MessageBuilderStrategy(ABC):
    """
    Clase abstracta que define el contrato para todos los constructores de mensajes.
    Utiliza un enfoque Data-Driven: recibe un layout pre-ordenado desde la Estrategia.
    """
    
    @abstractmethod
    def build_messages(self, system_prompt: str, layout: list[dict]) -> list:
        """
        'layout' es una lista ordenada de diccionarios creada por la Estrategia.
        Ejemplo:
        [
            {"type": "text", "content": "Busca un perro en estos frames:"},
            {"type": "image", "content": objeto_FramesPath_1},
            {"type": "image", "content": objeto_FramesPath_2}
        ]
        """
        pass

    def _format_base64_url(self, image_b64: str, mime_type: str = "image/jpeg") -> str:
        return f"data:{mime_type};base64,{image_b64}"


class CloudMessageBuilder(MessageBuilderStrategy):
    """
    Estrategia para modelos en la nube.
    Respeta la separación entre SystemMessage y HumanMessage.
    Traduce el layout genérico al formato específico de LangChain respetando el orden.
    """
    def build_messages(self, system_prompt: str, layout: list[dict]) -> list:

        human_content = []

        for item in layout:
            if item["type"] == "text":
                # Añade el bloque de texto donde la estrategia lo haya dictado
                human_content.append({"type": "text", "text": item["content"]})
            
            elif item["type"] == "image":
                # 1. Recuperamos el objeto FramesPath del layout
                frame_obj: FramesPath = item["content"]
                
                # 2. CONVERSIÓN LATE-BOUND (Lazy Evaluation): Leemos el archivo a Base64
                base64_img = encode_image_base64(frame_obj.frame_path)
                
                # 3. Lo metemos en el formato LangChain usando el frame_id
                human_content.append({"type": "text", "text": f"Frame {frame_obj.frame_id}:"})
                human_content.append({
                    "type": "image_url", 
                    "image_url": {"url": self._format_base64_url(base64_img)}
                })
        
        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content)
        ]


class LocalMessageBuilder(MessageBuilderStrategy):
    """
    Estrategia para modelos locales.
    Soluciona la "fuga de abstracción": a menudo ignoran el SystemMessage o 
    alucinan si hay múltiples roles. Se unifican las instrucciones de sistema
    y el layout del usuario en un único HumanMessage.
    """
    def build_messages(self, system_prompt: str, layout: list[dict]) -> list:

        # 1. Colocamos el System Prompt al principio del contenido "Humano"
        human_content = [{"type": "text", "text": f"INSTRUCCIONES DEL SISTEMA:\n{system_prompt}\n\n---"}]
        
        # 2. Iteramos sobre el layout que nos dictó la estrategia
        for item in layout:
            if item["type"] == "text":
                human_content.append({"type": "text", "text": f"Instrucción: {item['content']}"})
            
            elif item["type"] == "image":
                # 1. Recuperamos el objeto FramesPath del layout
                frame_obj: FramesPath = item["content"]
                
                # 2. CONVERSIÓN LATE-BOUND (Lazy Evaluation)
                base64_img = encode_image_base64(frame_obj.frame_path)
                
                # 3. Formato LangChain
                human_content.append({"type": "text", "text": f"Frame {frame_obj.frame_id}:"})
                human_content.append({
                    "type": "image_url", 
                    "image_url": {"url": self._format_base64_url(base64_img)}
                })
        
        return [HumanMessage(content=human_content)]