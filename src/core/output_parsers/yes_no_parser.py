import re
from src.core.output_parsers.base_parser import BaseFrameParser
from src.data.validators import FrameResults, FramesPath

class YesNoTextParser(BaseFrameParser):
    
    def get_format_instructions(self) -> str:
        return (
            "INSTRUCCIONES DE FORMATO: NO uses formato JSON. Responde en texto plano.\n"
            "- Si es una sola imagen: Responde empezando por 'SÍ' o 'NO', seguido de un guion y tu justificación.\n"
            "- Si es un lote de imágenes: Responde con exactamente una línea por imagen, siguiendo este patrón estricto:\n"
            "SÍ/NO - [Tu justificación aquí]\n"
            "No añadas saludos, ni introducciones, ni despidas el mensaje."
        )

    # ==========================================
    # MÉTODOS PÚBLICOS DE PARSEO
    # ==========================================

    def parse(self, text: str, frame_id: int) -> FrameResults:
        """Parseo de una única respuesta en texto libre (TemporalStrategy)."""
        texto_limpio = text.strip()
        
        # Extraemos el valor booleano y la justificación
        detectado = self._extraer_booleano(texto_limpio)
        descripcion = self._limpiar_descripcion(texto_limpio)
        
        return FrameResults(
            frame_id=frame_id, 
            detectado=detectado, 
            descripcion=descripcion
        )

    def parse_batch(self, text: str, lote: list[FramesPath]) -> list[FrameResults]:
        """Parseo de múltiples líneas de texto libre (BatchStrategy)."""
        # Separamos el texto por saltos de línea y quitamos líneas vacías
        lineas = [linea.strip() for linea in text.strip().split('\n') if linea.strip()]
        
        # Filtramos la "cháchara" del LLM para quedarnos solo con las líneas de respuesta
        lineas_utiles = self._filtrar_lineas_resultados(lineas, len(lote), text)
        
        resultados = []
        for frame_obj, linea in zip(lote, lineas_utiles):
            detectado = self._extraer_booleano(linea)
            descripcion = self._limpiar_descripcion(linea)
            
            resultados.append(FrameResults(
                frame_id=frame_obj.frame_id,
                detectado=detectado,
                descripcion=descripcion
            ))
            
        return resultados

    # ==========================================
    # MÉTODOS PRIVADOS DE APOYO (Clean Code)
    # ==========================================

    def _extraer_booleano(self, texto: str) -> bool:
        """
        Busca 'SÍ', 'SI', 'YES' o 'TRUE' al principio del texto.
        Solo miramos los primeros 15 caracteres para evitar falsos positivos
        si la palabra 'sí' aparece dentro de la justificación.
        """
        inicio_texto = texto[:15].upper()
        
        # Usamos RegEx para buscar la palabra exacta (evita que 'silla' cuente como 'sí')
        if re.search(r'\b(S[IÍ]|YES|TRUE)\b', inicio_texto):
            return True
        return False

    def _limpiar_descripcion(self, texto: str) -> str:
        """
        Limpia el 'SÍ - ' o 'NO - ' del principio para que la descripción quede elegante.
        Ej: 'SÍ - Hay un perro jugando' -> 'Hay un perro jugando'
        """
        # Elimina patrones como "Frame 1: ", "SÍ - ", "NO. ", etc. al inicio de la frase
        texto_limpio = re.sub(r'^(Frame\s*\d*:?\s*)?', '', texto, flags=re.IGNORECASE)
        texto_limpio = re.sub(r'^(S[IÍ]|NO|YES|TRUE|FALSE)\s*[-:.]?\s*', '', texto_limpio, flags=re.IGNORECASE)
        
        return texto_limpio.strip() if texto_limpio.strip() else "Sin descripción detallada."

    def _filtrar_lineas_resultados(self, lineas: list, tamano_esperado: int, texto_original: str) -> list:
        """
        Intenta cazar exactamente las líneas que corresponden a los frames,
        ignorando si el modelo dice "Aquí tienes los resultados:" al principio.
        """
        lineas_validas = []
        
        # Heurística 1: Buscar líneas que empiecen por nuestro patrón esperado
        for linea in lineas:
            if re.match(r'^(Frame|S[IÍ]|NO|YES|TRUE|FALSE)', linea, re.IGNORECASE):
                lineas_validas.append(linea)
                
        # Si la heurística 1 funcionó a la perfección:
        if len(lineas_validas) == tamano_esperado:
            return lineas_validas
            
        # Heurística 2 (Fallback): Si el modelo no usó el formato exacto pero escupió 
        # líneas sueltas y "cháchara" introductoria, nos quedamos con las últimas N líneas
        if len(lineas) >= tamano_esperado:
            return lineas[-tamano_esperado:]
            
        # Si no hay forma de cuadrarlo, lanzamos el error para forzar el reintento de la Strategy
        print(f"\n [DEBUG PARSER TEXTO] El LLM no devolvió suficientes líneas.\nTexto crudo:\n{texto_original}\n")
        raise ValueError(f"Se esperaban {tamano_esperado} respuestas, pero solo se detectaron {len(lineas)} líneas.")