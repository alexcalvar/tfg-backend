import os

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks

from core.pipeline import VLMPipeline
from core.model_manager import ModelManager

from utils.config_loader import ConfigLoader
from utils.file_utils import load_json, save_upload_file
from utils.project_status import ProjectStatus

from api.schemas import  HTTPResponse

app = FastAPI(title="API de Análisis VLM")
config = ConfigLoader()

@app.post("/api/v1/analyze", response_model=HTTPResponse, status_code=202)
async def analyze_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    user_prompt: str = Form(...),
    provider: str = Form(...), 
    model_name: str = Form(...)
):
    if not video.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser un formato de vídeo válido (mp4, avi, etc.).")

    try:
        modelo_solicitado, msg_strategy = ModelManager(provider, model_name).load_vlm()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al cargar el modelo: {str(e)}")

    pipeline = VLMPipeline(modelo_solicitado, provider, msg_strategy)
    video_path_destino = os.path.join(pipeline.video_dir, video.filename)
    
    await save_upload_file(video, video_path_destino)
    background_tasks.add_task(pipeline.process_video, video_path_destino, user_prompt)

    # Devolvemos usando el estándar
    return HTTPResponse(
        success=True,
        message="El archivo se ha guardado correctamente y el análisis ha comenzado.",
        data={
            "status": "processing",
            "project_id": os.path.basename(pipeline.base_run_dir),
            "video_file": video.filename
        }
    )


@app.get("/api/v1/status/{project_id}", response_model=HTTPResponse)
async def get_project_status(project_id: str):
    """Consulta si un proyecto está en cola, procesando o finalizado."""
    # Usamos consistentemente projects_folder (o result_folder, el que uses realmente)
    project_dir = config.get_path("projects_folder") 
    status_file_path = os.path.join(project_dir, project_id, "status.json")

    # Si no existe, es 404
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


@app.get("/api/v1/results/{project_id}", response_model=HTTPResponse)
async def get_project_results(project_id: str):
    """Devuelve el report.json final una vez que el vídeo ha sido procesado."""
    project_dir = config.get_path("projects_folder") # Misma clave que en /status
    project_path = os.path.join(project_dir, project_id)
    report_file_path = os.path.join(project_path, "results", "report.json")
    status_file_path = os.path.join(project_path, "status.json")

    if not os.path.exists(project_path):
        raise HTTPException(status_code=404, detail=f"El proyecto '{project_id}' no existe.")

    # Escudo para evitar UnboundLocalError
    state = "desconocido"
    if os.path.exists(status_file_path):
        try:
            status_file = load_json(status_file_path)
            state = status_file.get("state", "desconocido")
        except Exception:
            pass
    else:
        # F-string corregido
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
    

@app.get("/api/v1/providers") #como tal lo q pude ser util ver los proveedores porq los modelos unicamente hay q introducir el nombre 
async def list_available_models():
    """Lee el models_config.json y devuelve los modelos VLM que la API puede usar."""
    pass

@app.get("/api/v1/projects")
async def list_all_projects():
    """Devuelve una lista con todas las carpetas de proyectos históricos."""
    pass


@app.post("/api/v1/evaluate/{project_id}")
async def evaluate_project(project_id: str):
    """Ejecuta el BenchmarkRunner sobre un proyecto ya finalizado."""
    pass