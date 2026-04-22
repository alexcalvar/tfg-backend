import re
from src.core.output_parsers.base_parser import BaseFrameParser
from src.data.validators import FrameResults, FramesPath

from src.utils.logger import get_logger

logger = get_logger(__name__)

class YesNoTextParser(BaseFrameParser):
    
    def get_format_instructions(self) -> str:
        return (
            "INSTRUCCIONES DE FORMATO: NO uses formato JSON. Responde en texto plano.\n"
            "- Si es una sola imagen: Responde empezando por 'SÍ' o 'NO', seguido de un guion y tu justificación.\n"
            "- Si es un lote de imágenes: Responde con exactamente una línea por imagen, siguiendo este patrón estricto:\n"
            "SÍ/NO - [Tu justificación aquí]\n"
            "No añadas saludos, ni introducciones, ni despidas el mensaje."
        )



    def parse(self, text: str, frame_id: int) -> FrameResults:

        texto_limpio = text.strip()
        
        # extraemos el valor booleano y la justificación
        detectado = self._extraer_booleano(texto_limpio)
        descripcion = self._limpiar_descripcion(texto_limpio)
        
        return FrameResults(
            frame_id=frame_id, 
            detectado=detectado, 
            descripcion=descripcion
        )




    def parse_batch(self, text: str, lote: list[FramesPath]) -> list[FrameResults]:
        
        # separar el texto por saltos de línea y quitar líneas vacías
        lineas = [linea.strip() for linea in text.strip().split('\n') if linea.strip()]
        
        # filtrar solo las líneas de respuesta
        lineas_utiles = self._filtrar_lineas_resultados(lineas, len(lote), text)
        
        resultados = []
        for frame_obj, linea in zip(lote, lineas_utiles):
            detectado = self._extraer_booleano(linea)
            #descripcion = self._limpiar_descripcion(linea)
            
            resultados.append(FrameResults(
                frame_id=frame_obj.frame_id,
                detectado=detectado,
                descripcion=linea
            ))
            
        return resultados




    def _extraer_booleano(self, texto: str) -> bool:
        """
        Busca 'SÍ', 'SI', 'YES' o 'TRUE' al principio del texto.
        Solo mira los primeros 15 caracteres para evitar falsos positivos
        si la palabra 'sí' aparece dentro de la justificación.
        """
        inicio_texto = texto[:15].upper()
        
        # RegEx para buscar la palabra exacta, evita q palabras como silla cuenten como si
        if re.search(r'\b(S[IÍ]|YES|TRUE)\b', inicio_texto):
            return True
        return False




    def _limpiar_descripcion(self, texto: str) -> str:
        """
        Limpia el 'SÍ - ' o 'NO - ' del principio para que la descripción quede elegante.
        Ej: 'SÍ - Hay un perro jugando' -> 'Hay un perro jugando'
        """
        texto_limpio = re.sub(r'^(Frame\s*\d*:?\s*)?', '', texto, flags=re.IGNORECASE)
        texto_limpio = re.sub(r'^(S[IÍ]|NO|YES|TRUE|FALSE)\s*[-:.]?\s*', '', texto_limpio, flags=re.IGNORECASE)
        
        return texto_limpio.strip() if texto_limpio.strip() else "Sin descripción detallada."




    def _filtrar_lineas_resultados(self, lineas: list, tamano_esperado: int, texto_original: str) -> list:
        """
        Intenta seleccionar exactamente las líneas que corresponden a los frames,
        ignorando texto q el modelo pueda generar y q no sea parte de las respuestas sobre los frames
        """
        lineas_validas = []
        
        # primero buscar líneas que empiecen por el patrón esperado
        for linea in lineas:
            if re.match(r'^(Frame|S[IÍ]|NO|YES|TRUE|FALSE)', linea, re.IGNORECASE):
                lineas_validas.append(linea)
                
        # si la heurística 1 funcionó a la perfección:
        if len(lineas_validas) == tamano_esperado:
            return lineas_validas
            
        # en caso de no encontradas en primera instancia porque el modelo no usó el formato exacto pero escupió 
        # líneas sueltas y texto introductorio, coger las ultimas n lineas
        if len(lineas) >= tamano_esperado:
            logger.warning(f"El VLM añadió texto basura. Rescatando las últimas {tamano_esperado} líneas mediante heurística.")
            return lineas[-tamano_esperado:]
            
        # si no hay forma de arreglarlo, lanzar error para reintento del modleo
        logger.error(f"El VLM no devolvió suficientes líneas de respuesta.\nTexto crudo recibido:\n{texto_original}")
        raise ValueError(f"Se esperaban {tamano_esperado} respuestas, pero solo se detectaron {len(lineas)} líneas.")