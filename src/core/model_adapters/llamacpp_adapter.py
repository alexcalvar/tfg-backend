from typing import Any, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration

class CustomVisionLlamaCpp(BaseChatModel):
    """
    Adaptador personalizado para permitir a LangChain comunicarse con modelos multimodales locales de llama-cpp-python.
    """
    # Guardamos la instancia del modelo (en esta intancia se almacena el archivo del modelo con el archivo que lo dota de capacidad vsiual)
    cliente_nativo: Any 
    
    # instrucción para pydantic, de esta forma permite tipos externos como Llama y evitar q lance error de validación al intentar instanciar el adaptador
    class Config:
        arbitrary_types_allowed = True

    # identificador para langchain, al heredar de BaseChatModel, langchain necesita que este definido para poder identificar el modelo 
    @property
    def _llm_type(self) -> str:
        """Obligatorio en LangChain para telemetría/logging interno."""
        return "custom_vision_llamacpp"

    #se define el nuevo metodo _generate porque es el metodo interno q langchain llama cuanod se realiza un .invoke()
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        El corazón del adaptador. Aquí ocurre la magia de la traducción.
        """
        # --- LÓGICA DE TRADUCCIÓN ---
        # convertir 'messages' de formato langchain a formato llamacpp
        mensajes_llama = []


        for mensaje in messages:
            #  identificar la clase del objeto en la lista de mensajes para poder asignar el rol correspondiente 
            match mensaje:
                case SystemMessage():
                    rol = "system"
                case HumanMessage():
                    rol = "user"
                case AIMessage():
                    rol = "assistant"
                case _:
                    raise ValueError(f"Tipo de mensaje no soportado: {type(mensaje)}")
            
            contenido = mensaje.content 

            message_adapter = {"role": rol, "content": contenido}
            mensajes_llama.append(message_adapter)

        #  llamar a self.cliente_nativo.create_chat_completion(...), con esto se realiza una llamada a la libreria de C++ de llamacpp enviando la lista traducida
        #   al formato q espera llamacpp
        respuesta_cruda = self.cliente_nativo.create_chat_completion(
            messages=mensajes_llama,
            stop=stop, # se le pasan los tokens de parada si langchain los define
            **kwargs   # se pasa cualquier otra configuración extra (temperatura, n_ctx, ...), necesario para seguir el contrato de BaseChatModel
        )
        
        # proceso de empaquetado del resultado

        model_response = respuesta_cruda["choices"][0]["message"]["content"]

        #transformas el string de texto en un objeto de tipo mensaje de IA
        # importante porque LangChain diferencia entre lo que dice el humano (HumanMessage) y lo que responde la IA

        # envolver el mensaje en una capa de generacion, langchain usa esta clase para añadir información adicional si fuera necesario, 
        # ejempo, el motivo por el cual el modelo dejó de escribir 
        output_adapted = ChatGeneration(message=AIMessage(content=model_response))

        #chat results es el contenedor final que espera el framework.
        # langChain está diseñado para ser multirrespuesta entonces aunque solo haya una, el contrato exige que se devuelva dentro de una lista 
        # generations, para evitar errores de posición de pydantic
        return ChatResult(generations=[output_adapted])
        