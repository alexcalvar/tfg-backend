import os
import json
import base64

from fastapi import UploadFile

def ensure_dir(path: str):
    """Crea un directorio de forma segura si no existe."""
    if path:
        os.makedirs(path, exist_ok=True)

def load_json(file_path: str) -> dict:
    """Carga un archivo JSON garantizando la codificación UTF-8."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[ERROR CRÍTICO] No se encuentra el archivo JSON: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def save_json(data: list | dict, file_path: str):
    """Guarda un diccionario o lista en formato JSON de forma segura."""
    ensure_dir(os.path.dirname(file_path)) #asegurar q exista la carpeta
    
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def encode_image_base64(image_path: str) -> str:
    """Lee una imagen del disco y la convierte a cadena Base64."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[ERROR CRÍTICO] No se encuentra la imagen: {image_path}")
    
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
async def save_upload_file(upload_file: UploadFile, destination_path: str) -> str:
    """
    Guarda un archivo subido vía red en el disco duro por bloques (chunks).
    Evita la saturación de la memoria RAM con archivos pesados.
    """
    try:
        with open(destination_path, "wb") as buffer:
        # Leemos y escribimos en bloques de 1MB
            while content := await upload_file.read(1024 * 1024): 
                buffer.write(content)
        return destination_path
    except Exception as e:
        raise IOError(f"Error al guardar el archivo {upload_file.filename}: {str(e)}")
    finally:
        # resetear el puntero del archivo por si otra función necesita leerlo después
        await upload_file.seek(0)