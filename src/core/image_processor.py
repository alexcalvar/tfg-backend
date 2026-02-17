import base64 
import json
import os

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

class VLMProcessor:
    def __init__(self, model_instance):
         #Recibimos el modelo ya cargado desde el ModelManager
        self.vlm = model_instance
    

    def _encode_image(self, image_path):
        with open(image_path, "rb") as file:
            return base64.b64encode(file.read()).decode("utf-8")



    def analyze_frame( self,prompt, image_path):
        image_b64 = self._encode_image( image_path)

        human_msg = HumanMessage([{"type" : "text", "text" : f"{prompt}"},
                             {"type" : "image_url", "image_url" : f"data:image/jpeg;base64,{image_b64}"}])
        
        sys_msg = SystemMessage(content="""
                    Eres el encargado de analizar imagenes. Describe los objetos que ves en la imagen.
                    En caso de que algun objeto coincida con lo solicitado por el usuario responde true, en caso contrario, false            
                    Responde ÚNICAMENTE con un JSON válido siguiendo estrictamente este formato, sin texto adicional:
                    {
                        "detectado": true/false,
                        "descripcion": "Breve descripción de lo visualizado",
                        "confianza": "alto/medio/bajo"
                    }
                    """)
    
        try:

            response = self.vlm.invoke([sys_msg, human_msg]) #llamar al motor

            # response.content es un string, convertir a diccionario py
            resultado_json = json.loads(response.content)

            return resultado_json

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