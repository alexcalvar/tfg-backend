# 🎥 TFG Backend: Sistema de Análisis de Vídeo con IA Generativa (Zero-Shot)

> **Trabajo de Fin de Grado** - Escuela Superior de Ingeniería Informática (ESEI), Universidad de Vigo.
>
> **Autor:** Alejandro Calvar
> **Curso:** 2025/2026

## 📖 Descripción del Proyecto

Este repositorio contiene el **Backend** de una libreria para el análisis semántico de vídeo. El sistema utiliza estrategias de inferencia **Zero-Shot** mediante Modelos de Lenguaje Visual (VLM) para detectar eventos, generar descripciones y permitir búsquedas en lenguaje natural sobre contenidos audiovisuales, sin necesidad de entrenamiento previo de modelos específicos.

## 🚀 Características Arquitectónicas Principales
* **Procesamiento Asíncrono (`asyncio` / FastAPI):** Uso intensivo de `BackgroundTasks` y colas asíncronas (`asyncio.Queue`) para aislar la recepción de peticiones HTTP del procesamiento pesado de IA.
* **Extracción de Frames Tolerante a Fallos:** Motor de vídeo basado en `cap.grab()` de OpenCV que evita la corrupción de memoria y saltos ciegos en códecs comprimidos, garantizando una lectura secuencial perfecta.
* **Contratos Estrictos (Pydantic):** Todas las entradas y salidas de la API están matemáticamente tipadas usando esquemas de Pydantic (`StandardResponse[T]`), asegurando documentación automática (Swagger) y filtrado de seguridad en la salida de datos.
* **Máquina de Estados en Disco:** Seguimiento del progreso en tiempo real (`status.json`) guiado por enumerados estrictos (`ProjectStatus`), protegiendo el progreso de la IA ante posibles reinicios del servidor HTTP.
* **Configuración Dinámica:** Rutas, prompts de sistema y parámetros de vídeo inyectados dinámicamente mediante el patrón `ConfigLoader`.

## 📂 Estructura del Proyecto

```text
├── configs/
│   ├── config.json           # Rutas y parámetros generales del sistema
│   ├── models_config.json    # Catálogo de modelos VLM y proveedores
│   └── prompts.json          # Plantillas de sistema y usuario para la IA
├── datasets/                 # Almacenamiento local para pruebas (videos_test)
├── src/
│   ├── api/
│   │   ├── routes.py         # Controladores REST de FastAPI
│   │   └── schemas.py        # Modelos de validación Pydantic (StandardResponse)
│   ├── core/
│   │   ├── pipeline.py       # Orquestador principal (VLMPipeline)
│   │   ├── image_processor.py# Conector con los modelos VLM
│   │   ├── model_manager.py  # Gestor de dependencias de IA
│   │   
│   └── utils/
│       ├── file_utils.py     # Gestión de I/O de disco y subida por Chunks
│       ├── video_utils.py    # Motor de extracción OpenCV (VideoLoader)
│       └── config_loader.py  # Inyector de configuraciones
└── main.py                   # Script de ejecución en modo CLI local

