import os
import shutil

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks

from src.api.orchestrator import AnalysisOrchestrator
from src.data.enums import PostProcessingStr


from src.utils.config_loader import ConfigLoader
from src.utils.file_utils import load_json, get_list_models
from src.utils.project_status import ProjectStatus
from src.utils.logger import get_logger

from src.api.schemas import  HTTPResponse

endpoints = APIRouter()

config = ConfigLoader()

logger = get_logger(__name__)

@endpoints.post("/api/v1/events", response_model=HTTPResponse, status_code=202)
async def analyze_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    user_prompt: str = Form(...),
    vlm_provider: str = Form(...), 
    vlm_model_name: str = Form(...),
    processing_mode: str = Form(...),
    apply_alg : bool = Form(...)
):
    if not video.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser un formato de vídeo válido.")

    try:
        orchestrator = AnalysisOrchestrator()
        
        # delegamos el trabajo sucio al orquestador
        pipeline, project_id = await orchestrator.setup_video_pipeline(
            video_file=video,
            user_prompt=user_prompt,
            vlm_provider=vlm_provider,
            vlm_model=vlm_model_name,
            processing_mode=processing_mode,
            postprocessing_type=PostProcessingStr.ALGORITHM,
            apply_alg=apply_alg
        )
        
    except Exception as e:
        logger.exception("Error durante la orquestación")
        raise HTTPException(status_code=500, detail=f"Error al preparar el análisis: {str(e)}")

    background_tasks.add_task(pipeline.process_video, user_prompt)

    return HTTPResponse(
        success=True,
        message="El archivo se ha guardado correctamente y el análisis ha comenzado.",
        data={
            "status": "processing",
            "project_id": project_id,
            "video_file": video.filename
        }
    )



@endpoints.post("/api/v1/resums", response_model=HTTPResponse, status_code=202)
async def analyze_video_semantic(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    user_prompt: str = Form(...),
    vlm_provider: str = Form(...), 
    vlm_model_name: str = Form(...),
    llm_provider: str = Form(...), 
    llm_model_name: str = Form(...),
    processing_mode: str = Form(...)
):
    if not video.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser un formato de vídeo válido.")

    try:
        orchestrator = AnalysisOrchestrator()
        
        # delegamos el trabajo sucio al orquestador
        pipeline, project_id = await orchestrator.setup_video_pipeline(
            video_file=video,
            user_prompt=user_prompt,
            vlm_provider=vlm_provider,
            vlm_model=vlm_model_name,
            processing_mode=processing_mode,
            postprocessing_type=PostProcessingStr.SEMANTIC,
            llm_provider=llm_provider,
            llm_model=llm_model_name
        )
        
    except Exception as e:
        logger.exception("Error durante la orquestación")
        raise HTTPException(status_code=500, detail=f"Error al preparar el análisis: {str(e)}")

    # Lanzamos el proceso que ya tiene todo el contexto necesario
    background_tasks.add_task(pipeline.process_video, user_prompt)

    return HTTPResponse(
        success=True,
        message="El archivo se ha guardado correctamente y el análisis ha comenzado.",
        data={
            "status": "processing",
            "project_id": project_id,
            "video_file": video.filename
        }
    )



@endpoints.get("/api/v1/resums/{project_id}/status", response_model=HTTPResponse, status_code=200)
@endpoints.get("/api/v1/events/{project_id}/status", response_model=HTTPResponse, status_code=200)
def get_project_status(project_id: str):
    """Consulta si un proyecto está en cola, procesando o finalizado."""

    status_file_path = config.get_status_file_path(project_id=project_id)

    if not os.path.exists(status_file_path):
        raise HTTPException(status_code=404, detail=f"Archivo status.json no encontrado en el proyecto {project_id}.")

    try:
        status_data = load_json(status_file_path)
        return HTTPResponse(
            success=True,
            message="Estado recuperado correctamente.",
            data=status_data 
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Error interno en lectura del archivo status.json")    




@endpoints.get("/api/v1/events/{project_id}/results", response_model=HTTPResponse, status_code=200)
def get_events_results(project_id: str):
    """Devuelve el report.json final una vez que el vídeo ha sido procesado."""
   
    status_file_path = config.get_status_file_path(project_id=project_id)
    result_file_path = config.get_results_file_path(project_id, PostProcessingStr.ALGORITHM)

    if not os.path.exists(result_file_path):
        raise HTTPException(status_code=404, detail=f"El proyecto '{project_id}' no existe.")

    state = "desconocido"
    if os.path.exists(status_file_path):
        try:
            status_file = load_json(status_file_path)
            state = status_file.get("state", "desconocido")
        except Exception as e:
            logger.error(f"Error al leer status.json: {e}")
    
    else:
        raise HTTPException(status_code=404, detail=f"Archivo status.json no encontrado en el proyecto {project_id}, no se puede verificar el estado.")

    if state == ProjectStatus.COMPLETED.value:
        if os.path.exists(result_file_path):
            try:
                report_file = load_json(result_file_path)
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


@endpoints.get("/api/v1/resums/{project_id}/results", response_model=HTTPResponse, status_code=200)
def get_resums_results(project_id: str):
    """Devuelve el report.json final una vez que el vídeo ha sido procesado."""
   
    status_file_path = config.get_status_file_path(project_id=project_id)
    result_file_path = config.get_results_file_path(project_id, PostProcessingStr.SEMANTIC)

    if not os.path.exists(result_file_path):
        raise HTTPException(status_code=404, detail=f"El proyecto '{project_id}' no existe.")

    state = "desconocido"
    if os.path.exists(status_file_path):
        try:
            status_file = load_json(status_file_path)
            state = status_file.get("state", "desconocido")
        except Exception as e:
            logger.error(f"Error al leer status.json: {e}")
    
    else:
        raise HTTPException(status_code=404, detail=f"Archivo status.json no encontrado en el proyecto {project_id}, no se puede verificar el estado.")

    if state == ProjectStatus.COMPLETED.value:
        if os.path.exists(result_file_path):
            try:
                report_file = load_json(result_file_path)
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
def list_available_models():
    """Lee el models_config.json y devuelve los modelos VLM que la API puede usar."""
    
    config_folder_path = config.get_path("config_folder")

    models_config_path = os.path.join(config_folder_path, "models_config.json")

    models_list = get_list_models(models_config_path)

    return HTTPResponse(
        success=True,
        message="Lista de todos los modelos soportados por el sistema",
        data=models_list
    )



@endpoints.delete("/api/v1/{project_id}", response_model=HTTPResponse, status_code=200)
def delete_project(project_id: str):
    """ Borra un proyecto y todos sus archivos asociados de forma segura."""

    # prevención de path traversal
    # evitamos que alguien envíe "../../../" para intentar borrar el sistema operativo
    if ".." in project_id or "/" in project_id or "\\" in project_id:
        logger.warning(f"Intento de ataque de Path Traversal detectado. project_id: {project_id}")
        raise HTTPException(status_code=400, detail="ID de proyecto inválido. Contiene caracteres no permitidos.")
        
    # validación, rl sistema siempre usa el prefijo project_
    if not project_id.startswith("project_"):
        raise HTTPException(status_code=400, detail="Formato de ID de proyecto incorrecto.")

    # verificar q existe el proyecto
    projects_folder = config.get_path("projects_folder")
    project_path = os.path.join(projects_folder, project_id)

    if not os.path.exists(project_path):
        raise HTTPException(status_code=404, detail=f"El proyecto '{project_id}' no existe o ya ha sido eliminado.")

    # borrado recursivo
    try:
        shutil.rmtree(project_path)
        logger.info(f"Proyecto {project_id} y todos sus recursos eliminados correctamente del servidor.")
        
        return HTTPResponse(
            success=True,
            message=f"El proyecto {project_id} ha sido eliminado correctamente.",
            data=None
        )
    except Exception as e:
        logger.error(f"Error crítico al intentar eliminar el directorio del proyecto {project_id}: {e}")
        raise HTTPException(status_code=500, detail="Error interno al intentar liberar los recursos del servidor.")
