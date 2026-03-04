import os

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from typing import Optional

from core.pipeline import VLMPipeline
from core.model_manager import ModelManager

from utils.file_utils import save_upload_file

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hola desde mi primera API con FastAPI"}

@app.post("/api/v1/analyze")
async def analyze_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    user_prompt: str = Form(...),
    provider: str = Form(...), 
    model_name: str = Form(...)
):
    #  Cargar dependencias del modelo
    modelo_solicitado, msg_strategy = ModelManager(provider, model_name).load_vlm()

    #  Instanciar el orquestador 
    pipeline = VLMPipeline(modelo_solicitado, provider, msg_strategy)

    # Guardar el archivo físicamente 
    video_path_destino = os.path.join(pipeline.video_dir, video.filename)
    await save_upload_file(video, video_path_destino)

    # Enviar a la cola de procesamiento en segundo plano
    background_tasks.add_task(pipeline.process_video, video_path_destino, user_prompt)

    # Respuesta  al cliente
    return {
        "status": "processing",
        "project_id": os.path.basename(pipeline.base_run_dir),
        "message": "El archivo se ha guardado correctamente y el análisis ha comenzado.",
        "video_file": video.filename
    }


@app.get("/api/v1/analyze/{id}")
def analyze_status():
    
    return

@app.get("/api/v1/analyze/{id}/result")
def return_result():
    
    return
