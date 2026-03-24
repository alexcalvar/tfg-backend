import os

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks

from src.core.pipeline import VLMPipeline
from src.core.model_factory import ModelFactory
from src.core.processing_factory import ProcessingFactory

from src.utils.config_loader import ConfigLoader
from src.utils.file_utils import load_json, save_upload_file, get_list_models
from src.utils.project_status import ProjectStatus

from src.api.schemas import  HTTPResponse

endpoints = APIRouter()

config = ConfigLoader()

@endpoints.post("/api/v1/analyze", response_model=HTTPResponse, status_code=202)
async def analyze_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    user_prompt: str = Form(...),
    provider: str = Form(...), 
    model_name: str = Form(...),
    processing_mode: str = Form(...)
):
    if not video.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser un formato de vídeo válido.")

    try:
        modelo_solicitado, msg_strategy = ModelFactory().load_vlm(provider, model_name)
        processing_strategy = ProcessingFactory.create_strategy(processing_mode)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al cargar el modelo: {str(e)}")

    pipeline = VLMPipeline(modelo_solicitado, provider, msg_strategy, processing_strategy)
    
    video_path_destino = os.path.join(pipeline.video_dir, video.filename)
    
    await save_upload_file(video, video_path_destino)   #tambien se almacena el video en la carpeta del proyecto correspondiente

    background_tasks.add_task(pipeline.process_video, video_path_destino, user_prompt)

    return HTTPResponse(
        success=True,
        message="El archivo se ha guardado correctamente y el análisis ha comenzado.",
        data={
            "status": "processing",
            "project_id": os.path.basename(pipeline.base_run_dir),
            "video_file": video.filename
        }
    )


@endpoints.get("/api/v1/status/{project_id}", response_model=HTTPResponse, status_code=200)
async def get_project_status(project_id: str):
    """Consulta si un proyecto está en cola, procesando o finalizado."""

    project_dir = config.get_path("projects_folder") 
    status_file_path = os.path.join(project_dir, project_id, "status.json")

    if not os.path.exists(status_file_path):
        raise HTTPException(status_code=404, detail=f"Archivo status.json no encontrado en el proyecto {project_id}.")

    try:
        status_data = load_json(status_file_path)
        return HTTPResponse(
            success=True,
            message="Estado recuperado correctamente.",
            data=status_data 
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error interno en lectura del archivo status.json")    


@endpoints.get("/api/v1/results/{project_id}", response_model=HTTPResponse, status_code=200)
async def get_project_results(project_id: str):
    """Devuelve el report.json final una vez que el vídeo ha sido procesado."""
   
    project_dir = config.get_path("projects_folder") 
    project_path = os.path.join(project_dir, project_id)
    report_file_path = os.path.join(project_path, "results", "report.json")
    status_file_path = os.path.join(project_path, "status.json")

    if not os.path.exists(project_path):
        raise HTTPException(status_code=404, detail=f"El proyecto '{project_id}' no existe.")

    state = "desconocido"
    if os.path.exists(status_file_path):
        try:
            status_file = load_json(status_file_path)
            state = status_file.get("state", "desconocido")
        except Exception:
            pass
    
    else:
        raise HTTPException(status_code=404, detail=f"Archivo status.json no encontrado en el proyecto {project_id}, no se puede verificar el estado.")

    if state == ProjectStatus.COMPLETED.value:
        if os.path.exists(report_file_path):
            try:
                report_file = load_json(report_file_path)
                return HTTPResponse(
                    success=True,
                    message="Resultados obtenidos correctamente.",
                    data=report_file
                )
            except Exception as e: 
                raise HTTPException(status_code=500, detail=f"Error interno al leer el informe de resultados: {str(e)}")
        else:
            raise HTTPException(status_code=500, detail=f"No se encontro el archivo report.json en el proyecto {project_id}")
    else:
        raise HTTPException(
            status_code=400, 
            detail=f"Los resultados aún no están listos. El estado actual del proyecto es: '{state}'."
        )
    

@endpoints.get("/api/v1/providers", response_model=HTTPResponse, status_code=200) 
async def list_available_models():
    """Lee el models_config.json y devuelve los modelos VLM que la API puede usar."""
    
    config_folder_path = config.get_path("config_folder")

    models_config_path = os.path.join(config_folder_path, "models_config.json")

    models_list = get_list_models(models_config_path)

    return HTTPResponse(
        success=True,
        message="Lista de todos los modelos soportados por el sistema",
        data=models_list
    )

