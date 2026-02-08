# 🎥 TFG Backend: Sistema de Análisis de Vídeo con IA Generativa (Zero-Shot)

> **Trabajo de Fin de Grado** - Escuela Superior de Ingeniería Informática (ESEI), Universidad de Vigo.
>
> **Autor:** [Alejandro Calvar]
> **Tutor/a:** [Martín Pérez]
> **Curso:** 2025/2026

## 📖 Descripción del Proyecto

Este repositorio contiene el **Backend** de una plataforma modular para el análisis semántico de vídeo. El sistema utiliza estrategias de inferencia **Zero-Shot** mediante Modelos de Lenguaje Visual (VLM) para detectar eventos, generar descripciones y permitir búsquedas en lenguaje natural sobre contenidos audiovisuales, sin necesidad de entrenamiento previo de modelos específicos.

El núcleo del sistema implementa un pipeline de **6 fases** que transforma datos visuales no estructurados (píxeles) en información estructurada (JSON), priorizando la tolerancia a fallos y la eficiencia en hardware de consumo.

## 🚀 Stack Tecnológico

El proyecto está construido sobre una arquitectura moderna y asíncrona:

* **Lenguaje:** Python 3.11+
* **API Framework:** FastAPI (Uvicorn server)
* **Visión por Computador:** OpenCV (cv2)
* **Orquestación IA:** LangChain 
* **Inferencia Local:** Ollama 
* **Validación de Datos:** Pydantic

## ⚙️ Arquitectura del Sistema

El backend sigue una arquitectura limpia (Clean Architecture) dividida en capas:

```text
src/
├── api/            # Controladores y Endpoints (REST)
├── core/           # Lógica de Negocio (Pipeline, Algoritmia de Normalización)
├── llm/            # Capa de Abstracción de IA (Conectores LangChain)
├── schemas/        # Modelos de Datos (Pydantic / DTOs)
└── main.py         # Punto de entrada de la aplicación
data/               # Almacenamiento temporal (Frames, Uploads, JSONs)