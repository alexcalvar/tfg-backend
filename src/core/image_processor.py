import base64 
import json
import os
import asyncio

from langchain_core.language_models.chat_models import BaseChatModel
from core.adapters.message_builders import MessageBuilderStrategy
from data.validators import FrameAnalysisResult

class VLMProcessor:
    def __init__(self, vlm_model_instance : BaseChatModel, selected_message_strategy : MessageBuilderStrategy, system_prompt, task_template):
        self.vlm = vlm_model_instance
        self.message_strategy = selected_message_strategy
        self.sys_prompt = system_prompt
        self.task_template = task_template 

    def _encode_image(self, image_path):
        with open(image_path, "rb") as file:
            return base64.b64encode(file.read()).decode("utf-8")
        
    def _clean_json_string(self, text: str) :   
        """Limpia el output del LLM para asegurar que es parseable por json.loads"""
        text = text.strip()
        # Si el modelo responde con bloques de markdown, se limpian
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        # Buscar el primer y último corchete por si hay texto de chat antes o después
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            return text[start_idx:end_idx+1]
        return text



    def  analyze_frame( self,user_prompt, image_path):
        image_b64 = self._encode_image( image_path)

        # Metemos la pregunta del usuario dentro de nuestra plantilla estricta
        prompt_formateado = self.task_template.replace("{user_query}", user_prompt)

        # Le pasamos el prompt_formateado a la estrategia en vez de la pregunta cruda
        final_prompt = self.message_strategy.build_messages(self.sys_prompt, prompt_formateado, image_b64)
    
        try:
            if False : 
                vlm_estricto = self.vlm.with_structured_output(FrameAnalysisResult)
                frame_response = vlm_estricto.invoke(final_prompt) #llamar al motor con salida definida
                return frame_response.model_dump()
            
            else :
                response = self.vlm.invoke(final_prompt)
                clear_response = self._clean_json_string(response.content)
                return json.loads(clear_response)

        except Exception as e:
            print(f"Error analizando {e}")
            return {"error": str(e)}
        

# --- BLOQUE DE PRUEBA ---
if __name__ == "__main__": 
    print("--- INICIANDO PRUEBA DEL CONECTOR ---")
    connector = VLMProcessor()
    
    # pedir ruta y quitar comillas por si el usuario arrastra el archivo
    ruta = input("Por favor, introduce la ruta de la imagen: ").strip('"').strip("'") 

    if os.path.exists(ruta):
        resultado = connector.analyze_frame("dime si en la imagen aparece un coche azul",ruta) 
        print("\n RESULTADO FINAL:")
        print(json.dumps(resultado, indent=4, ensure_ascii=False))
    else:
        print(" El archivo no existe. Revisa la ruta.")