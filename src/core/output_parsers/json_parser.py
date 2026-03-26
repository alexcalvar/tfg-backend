import json
from src.core.output_parsers.base_parser import BaseFrameParser
from src.data.validators import FrameResults, FramesPath

class JsonFrameParser(BaseFrameParser):
    
    def get_format_instructions(self) -> str:
        return (
            "INSTRUCCIONES DE FORMATO: Responde ÚNICA Y EXCLUSIVAMENTE con JSON válido. "
            "No incluyas texto explicativo. No uses bloques de código markdown (```json). "
            "Para un fotograma, devuelve un objeto JSON. Para varios fotogramas, devuelve una lista de objetos JSON. "
            "Claves obligatorias: 'detectado' (booleano) y 'descripcion' (string)."
        )



    def parse(self, text: str, frame_id: int) -> FrameResults:

        diccionario = self._decodificar_json_seguro(text)
        
        #comprobacion de si el vlm respondio con una lista
        if isinstance(diccionario, list) and len(diccionario) > 0:
            diccionario = diccionario[0]

        if not isinstance(diccionario, dict):
             raise ValueError(f"Se esperaba un objeto JSON (dict), pero se recibió: {type(diccionario)}")
            
        return FrameResults(
            frame_id=frame_id, 
            detectado=self._normalizar_booleano(diccionario.get("detectado", False)), 
            descripcion=diccionario.get("descripcion", "Error: Sin descripción")
        )




    def parse_batch(self, text: str, lote: list[FramesPath]) -> list[FrameResults]:

        estructura_cruda = self._decodificar_json_seguro(text)
        
        # revisa la estructura devuelta por el vlm, en caso de que haya algun error intenta 
        # arreglar la estructura para continuar con el parseado
        lista_diccionarios = self._salvar_estructura_diccionario(estructura_cruda)

        # validaciones
        self._validar_lote_final(lista_diccionarios, len(lote), text)
        
        # mapear a frame result 
        resultados = []

        for frame_obj, dict_respuesta in zip(lote, lista_diccionarios):
        
            resultados.append(FrameResults(
                frame_id=frame_obj.frame_id,
                detectado=self._normalizar_booleano(dict_respuesta.get("detectado", False)), #en caso de algun problema con el valor de detectado , guardar false
                descripcion=dict_respuesta.get("descripcion", "Error: Sin descripción")
            )
        )
            
        return resultados

 
 

    def _decodificar_json_seguro(self, text: str):
        """Limpia la basura markdown y decodifica el JSON."""
        # Limpieza básica de Markdown
        texto_limpio = text.strip()
        if texto_limpio.startswith("```json"):
            texto_limpio = texto_limpio[7:]
        elif texto_limpio.startswith("```"):
            texto_limpio = texto_limpio[3:]
        
        if texto_limpio.endswith("```"):
            texto_limpio = texto_limpio[:-3]
            
        texto_limpio = texto_limpio.strip()

        try:
            return json.loads(texto_limpio)
        except json.JSONDecodeError as e:
            print(f"\n [DEBUG PARSER] El VLM no devolvió JSON válido:\n{texto_limpio}\n")
            raise ValueError(f"Error decodificando JSON: {e}")




    def _salvar_estructura_diccionario(self, estructura):
        """Aplica heurísticas para corregir fallos comunes de formato de los VLMs."""
        if not isinstance(estructura, dict):
            return estructura # Si ya es lista, no hacemos nada

        # envolvió la lista en una clave -> {"resultados": [...]}
        for key, value in estructura.items():
            if isinstance(value, list):
                return value
        
        # devolvió un diccionario de diccionarios, ej: {"frame1": {...}, "frame2": {...}}
        #if all(isinstance(v, dict) for v in estructura.values()):
        #    return list(estructura.values())
            
        #  devolvió 1 solo objeto asumiendo que aplica a todo el lote
        if "detectado" in estructura:
            return [estructura] 

        return estructura # Devolvemos lo que haya si no pudimos salvarlo




    def _validar_lote_final(self, lista_diccionarios, tamano_esperado: int, texto_original: str):
        """Asegura que la estructura resultante es válida para empaquetarla."""
        
        if not isinstance(lista_diccionarios, list):
            print(f"\n [ERROR PARSER] Estructura devuelta por el VLM no soportada:\n{texto_original}\n")
            raise ValueError(f"El VLM no devolvió una lista válida. Recibido: {type(lista_diccionarios)}")
            
        if len(lista_diccionarios) != tamano_esperado:
            print(f"\n [ERROR PARSER] Desajuste de tamaño. VLM devolvió:\n{texto_original}\n")
            raise ValueError(f"Desajuste: El modelo devolvió {len(lista_diccionarios)} resultados, pero el lote tiene {tamano_esperado} imágenes.")




    def _normalizar_booleano(self, valor) -> bool:
        """asegurar que las variantes de TRUE se interpreten como booleano"""
        if isinstance(valor, str):
            return valor.strip().lower() == "true"
        return bool(valor)