from abc import ABC, abstractmethod
from langchain_core.messages import HumanMessage, SystemMessage

from src.data.validators import FramesPath
from src.utils.file_utils import encode_image_base64

class MessageStrategy(ABC):
    """
    Clase abstracta que define el contrato para todos los constructores de mensajes,
     recibe un layout desde la estrategia que define el formato de la peticion que se enviara al modelo
    """
    
    @abstractmethod
    def build_messages(self, system_prompt: str, layout: list[dict]) -> list:
        """
        Funcion encargada de generar las peticiones que se enviaran al modelo siguiendo el formato 
        del layout definido para la estrategia de procesamiento correspondiente
        """
        pass

    def _format_base64_url(self, image_b64: str, mime_type: str = "image/jpeg") -> str:
        return f"data:{mime_type};base64,{image_b64}"


class CloudMessageBuilder(MessageStrategy):
    """
    Estrategia para modelos en la nube. Respeta la separación entre SystemMessage y HumanMessage.
    Traduce el layout genérico al formato específico respetando el orden.
    """
    def build_messages(self, system_prompt: str, layout: list[dict]) -> list:

        human_content = []

        for item in layout:
            if item["type"] == "text":
                # Añade el bloque de texto donde la estrategia lo haya dictado
                human_content.append({"type": "text", "text": item["content"]})
            
            elif item["type"] == "image":
                # recuperar el objeto FramesPath del layout
                frame_obj: FramesPath = item["content"]
                
                # conversion leer el archivo a B64 (lazy evaluation, incluir en la docu)
                base64_img = encode_image_base64(frame_obj.frame_path)
                
                # incluir en el formato usando el frame_id
                human_content.append({"type": "text", "text": f"Frame {frame_obj.frame_id}:"})
                human_content.append({
                    "type": "image_url", 
                    "image_url": {"url": self._format_base64_url(base64_img)}
                })
        
        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content)
        ]


class LocalMessageBuilder(MessageStrategy):
    """
    Estrategia para modelos locales, a menudo ignoran el SystemMessage o 
    alucinan si hay múltiples roles por lo tanto se unifican las instrucciones de sistema
    y el layout del usuario en un unico HumanMessage.
    """
    def build_messages(self, system_prompt: str, layout: list[dict]) -> list:

        # system prompt al principio del contenido 
        human_content = [{"type": "text", "text": f"INSTRUCCIONES DEL SISTEMA:\n{system_prompt}\n\n---"}]
        
        # iterar sobre el layout determinado para la estrategia seleccionada
        for item in layout:
            if item["type"] == "text":
                human_content.append({"type": "text", "text": f"Instrucción: {item['content']}"})
            
            elif item["type"] == "image":
                # recuperar el framepath del layout
                frame_obj: FramesPath = item["content"]
                
                # conversion
                base64_img = encode_image_base64(frame_obj.frame_path)
                
                # par frame-img en b64
                human_content.append({"type": "text", "text": f"Frame {frame_obj.frame_id}:"})
                human_content.append({
                    "type": "image_url", 
                    "image_url": {"url": self._format_base64_url(base64_img)}
                })
        
        return [HumanMessage(content=human_content)]