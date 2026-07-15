# 🎥 TFG Backend: Sistema de Análisis de Vídeo con IA Generativa (Zero-Shot)

> **Trabajo de Fin de Grado** - Escuela Superior de Ingeniería Informática (ESEI), Universidad de Vigo.
>
> **Autor:** Alejandro Calvar
> **Curso:** 2025/2026

## 📖 Descripción del proyecto

Este repositorio contiene el **backend** de un sistema para el análisis semántico de vídeo mediante **Modelos de Lenguaje Visual (VLM)** en modo **Zero-Shot**, es decir, sin entrenamiento previo sobre tareas específicas. A partir de una consulta en lenguaje natural del usuario, el sistema ofrece dos funcionalidades principales sobre un mismo pipeline de extracción y análisis de fotogramas:

- **Detección de eventos**: identifica en qué intervalos de tiempo del vídeo ocurre lo que el usuario ha descrito (p. ej. *"dime cuándo aparece un perro en la escena"*), aplicando opcionalmente un algoritmo de postprocesado que suaviza detecciones ruidosas.
- **Resumen narrativo (semántico)**: genera un resumen jerárquico del contenido del vídeo, enfocado según la consulta del usuario, combinando las descripciones de cada fotograma en niveles sucesivos mediante un LLM.

La API se expone vía **FastAPI**, es totalmente **asíncrona** (los análisis se ejecutan en segundo plano y se pueden consultar, cancelar o eliminar mediante `project_id`) y admite múltiples proveedores de modelos, tanto locales (`llamacpp`, `ollama`) como en la nube (`google`, `openroute`, `groq`).

## 🚀 Características arquitectónicas principales

* **Procesamiento asíncrono productor-consumidor** (`asyncio` / FastAPI `BackgroundTasks`): la extracción de fotogramas y su análisis por el VLM corren en paralelo mediante una `asyncio.Queue`, desacoplando la recepción HTTP del trabajo pesado de IA.
* **Cancelación cooperativa**: cada proyecto en ejecución tiene asociado un `asyncio.Event` gestionado por un registro centralizado (`TaskRegistry`), consultado en puntos de control durante la extracción y el análisis para permitir detener un proceso en curso de forma limpia.
* **Extracción de fotogramas tolerante a fallos**: motor de vídeo basado en OpenCV (`VideoLoader`) con lectura secuencial, redimensionado configurable y reintentos por fotograma.
* **Patrón Strategy** para intercambiar en tiempo de ejecución tanto la forma de agrupar fotogramas al enviarlos al VLM (`BatchStrategy` / `TemporalStrategy`) como el algoritmo de postprocesado (detección de eventos vía `SlidingWindowNormalizer` / `StateLockNormalizer`, o resumen semántico vía `SemanticAnalyzer`).
* **Patrón Factory** (`ModelFactory`, `ProcessingFactory`, `AlgorithmFactory`) para resolver de forma dinámica, a partir de configuración, qué proveedor de modelo, estrategia de procesamiento o algoritmo de postprocesado se instancia en cada petición.
* **Patrón Observer**: las estrategias de procesamiento notifican su progreso a un `ProjectStatusManager`, que persiste el estado (`status.json`) de forma que sobrevive a un reinicio del servidor.
* **Contratos estrictos con Pydantic**: todas las entradas y salidas de la API están tipadas (`HTTPResponse`, `FrameResults`, `EventInterval`, `SummaryNode`...), lo que habilita documentación automática (Swagger) y validación de datos.
* **Configuración dinámica** (`ConfigLoader`, patrón *singleton*): rutas, prompts de sistema y parámetros de vídeo/postprocesado se leen de `config.properties` y `configs/*.json` sin necesidad de tocar código.

## 📂 Estructura del proyecto

```text
├── configs/
│   ├── models_config.json        # Catálogo de modelos VLM/LLM y proveedores soportados
│   └── prompts.json               # Plantillas de sistema y de tarea para el VLM/LLM
├── config.properties              # Rutas, parámetros de vídeo y de postprocesado
├── datasets/
│   ├── videos_test/                # Vídeos de prueba para el CLI y los benchmarks
│   └── benchmarks/                 # Ground truth y reportes de evaluación
├── docs/
│   └── manual_usuario.md          # Manual de usuario detallado (API, interfaz web, configuración)
├── projects/                       # Carpeta de trabajo por proyecto (vídeo, frames, resultados, status.json)
├── scripts/
│   ├── benchmarks.py               # Batería automatizada de experimentos VLM/estrategia
│   └── evaluate_results.py         # Cálculo de métricas sobre resultados ya generados
├── src/
│   ├── api/
│   │   ├── routes.py                # Endpoints REST de FastAPI
│   │   ├── orchestrator.py          # Ensamblaje del pipeline a partir de los parámetros de la petición
│   │   ├── task_registry.py         # Registro (singleton) de tokens de cancelación por proyecto
│   │   └── schemas.py               # Modelos Pydantic de entrada/salida
│   ├── core/
│   │   ├── pipeline.py               # VLMPipeline: orquestador principal del análisis
│   │   ├── image_processor.py        # Conector entre las estrategias y el modelo VLM
│   │   ├── factories/                # ModelFactory, ProcessingFactory, AlgorithmFactory
│   │   ├── frame_providers/          # VideoLoader: extracción de fotogramas con OpenCV
│   │   ├── processing_strategies/    # BatchStrategy y TemporalStrategy
│   │   ├── output_parsers/           # Parseo de la respuesta del VLM (JSON / texto sí-no)
│   │   ├── message_strategies/       # Construcción de mensajes multimodales por proveedor
│   │   └── model_adapters/           # Adaptadores para modelos locales (llama.cpp)
│   ├── postprocessing/
│   │   ├── postprocessing_algorithms/# SlidingWindowNormalizer, StateLockNormalizer
│   │   └── resums_logic/             # SemanticAnalyzer (árbol de resumen narrativo)
│   ├── observer/                     # ProjectStatusManager y contrato StatusObserver
│   ├── evaluation/                   # Cálculo de métricas y generación de reportes de benchmark
│   ├── data/                          # Enums, validadores Pydantic, carga de datasets
│   └── utils/                         # ConfigLoader, logger, utilidades de ficheros
├── tests/                             # Batería de pruebas unitarias (pytest)
├── local_tester.py                    # Cliente CLI interactivo para probar el pipeline sin la API
├── webapp.py                          # Interfaz web de demostración (Gradio) que consume la API
└── main.py                            # Punto de entrada de la API (FastAPI)
```

## 🔧 Instalación

Requisitos: Python 3.10+.

```bash
# Crear y activar un entorno virtual
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux / macOS

# Instalar dependencias
pip install -r requirements.txt
```

Configura tus credenciales en un archivo `.env` en la raíz del proyecto (solo son necesarias las de los proveedores que vayas a usar):

```env
OLLAMA_BASE_URL=http://localhost:11434
GOOGLE_API_KEY=tu_api_key_de_google
GROQ_API_KEY=tu_api_key_de_groq
OPEN_ROUTE_API_KEY=tu_api_key_de_openrouter
```

Los modelos disponibles (y sus parámetros) se declaran en `configs/models_config.json`; el comportamiento general del pipeline (intervalo de extracción, algoritmo de postprocesado, prompts activos, etc.) se ajusta en `config.properties`.

## ▶️ Puesta en marcha

**API REST** (documentación interactiva Swagger en `http://localhost:8000/docs`):

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Interfaz web de demostración** (con la API ya arrancada, en otra terminal):

```bash
python webapp.py
```

**Cliente CLI interactivo**, para probar el pipeline por terminal sin pasar por la API:

```bash
python local_tester.py
```

Consulta el **[Manual de Usuario](docs/manual_usuario.md)** para el detalle completo de cada endpoint, los parámetros aceptados, el formato de los resultados y la resolución de problemas habituales.

## ✅ Testing

```bash
pytest
```

El proyecto incluye una batería de pruebas unitarias (`tests/`) que cubren, de forma aislada mediante mocks, los endpoints de la API, las factorías de modelos, la extracción de fotogramas, los parsers de salida del VLM, las estrategias de procesamiento, el pipeline completo y los algoritmos de postprocesado.

