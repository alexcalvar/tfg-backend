import os
import json
import base64

from fastapi import UploadFile

def ensure_dir(path: str):
    """crea un directorio de forma segura si no existe"""
    if path:
        os.makedirs(path, exist_ok=True)



def load_json(file_path: str) -> dict:
    """carga un archivo json con codificación utf-8"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[ERROR] No se encuentra el archivo JSON: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)



def save_json(data: list | dict, file_path: str):
    """guarda un diccionario o lista en formato json"""
    ensure_dir(os.path.dirname(file_path)) #asegurar q exista la carpeta
    
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)



def save_results(data, file_path: str):
    """
    Guarda datos en un archivo JSON. Es capaz de procesar listas de modelos Pydantic, 
    modelos individuales o diccionarios nativos.
    """
    # 1. Aseguramos que la carpeta existe
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # 2. Detectamos qué tipo de dato nos han enviado
    if isinstance(data, list) and len(data) > 0 and hasattr(data[0], "model_dump"):
        # Es una lista de Pydantic models (Ej: FrameResults de la fase visual)
        datos_a_guardar = [item.model_dump() for item in data]
        
    elif hasattr(data, "model_dump"):
        # Es un único Pydantic model (Ej: SummaryNode si no le hiciéramos dump antes)
        datos_a_guardar = data.model_dump()
        
    else:
        # Ya es un diccionario o una lista normal (Ej: root.model_dump() de tu SemanticAnalyzer)
        datos_a_guardar = data

    # 3. Guardamos el archivo
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(datos_a_guardar, f, indent=4, ensure_ascii=False)



def encode_image_base64(image_path: str) -> str:
    """lee una imagen del disco y la convierte a cadena base64"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[ERROR] No se encuentra la imagen: {image_path}")
    
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    


async def save_upload_file(upload_file: UploadFile, destination_path: str) -> str:
    """
    guarda un archivo subido en el disco duro por bloques
    """
    try:
        with open(destination_path, "wb") as buffer:
        # leemos y escribimos en bloques 
            while content := await upload_file.read(1024 * 1024): 
                buffer.write(content)
        return destination_path
    except Exception as e:
        raise IOError(f"Error al guardar el archivo {upload_file.filename}: {str(e)}")
    finally:
        # resetear el puntero del archivo por si otra función necesita leerlo después
        await upload_file.seek(0)



def get_list_models(config_path: str) -> dict:
    """
    Devuelve los proveedores y los modelos q soportan
    """
    try:
        config = load_json(config_path)
        vlms_config = config.get("vlms", {})
        
        estructura = {
            proveedor: list(modelos.keys()) 
            for proveedor, modelos in vlms_config.items()
        }
        
        return estructura

    except Exception as e:
        print(f"Error al obtener la estructura de proveedores: {e}")
        return {}        