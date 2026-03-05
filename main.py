from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI

from src.api.routes import endpoints 

# instanciar api
app = FastAPI(
    title="API de Análisis VLM",
    description="Orquestador asíncrono para modelos de visión-lenguaje (TFG)",
    version="1.0.0"
)

#vincular endpoints
app.include_router(endpoints)

