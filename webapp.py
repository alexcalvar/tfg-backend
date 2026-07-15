import os
import time
import mimetypes

import requests
import gradio as gr

from src.data.enums import StrategyType

DEFAULT_API_URL = "http://127.0.0.1:8000"
POLL_INTERVAL_SECONDS = 3

TIPO_EVENTOS = "Detección de eventos"
TIPO_RESUMEN = "Resumen semántico"

ESTRATEGIAS = {estrategia.name: estrategia.value for estrategia in StrategyType}
ESTADOS_TERMINALES = ("completed", "error", "canceled")

VIDEO_PLAYER_ELEM_ID = "video_player"


def _extraer_detalle_error(respuesta: requests.Response) -> str:
    try:
        return respuesta.json().get("detail", respuesta.text)
    except ValueError:
        return respuesta.text


def _endpoint_analisis(tipo_analisis: str) -> str:
    return "/api/v1/events" if tipo_analisis == TIPO_EVENTOS else "/api/v1/resums"


def _endpoint_status(project_id: str) -> str:
   
    return f"/api/v1/{project_id}/status"


def _endpoint_resultados(tipo_analisis: str, project_id: str) -> str:
    prefijo = "events" if tipo_analisis == TIPO_EVENTOS else "resums"
    return f"/api/v1/{prefijo}/{project_id}/results"


def _formatear_tiempo(segundos: float) -> str:
    segundos_totales = int(segundos)
    horas, resto = divmod(segundos_totales, 3600)
    minutos, segs = divmod(resto, 60)
    if horas:
        return f"{horas:02d}:{minutos:02d}:{segs:02d}"
    return f"{minutos:02d}:{segs:02d}"


def _construir_html_eventos(intervalos: list) -> str:
    """Genera enlaces que, al pulsarlos, mueven el reproductor de vídeo al inicio de cada intervalo."""
    if not intervalos:
        return ""

    filas = []
    for evento in intervalos:
        inicio = evento.get("start_timestamp", 0)
        fin = evento.get("end_timestamp", 0)
        etiqueta = f"Evento {evento.get('event_id')}: {_formatear_tiempo(inicio)} – {_formatear_tiempo(fin)}"
        filas.append(
            '<a href="javascript:void(0)" '
            f"onclick=\"var v=document.querySelector('#{VIDEO_PLAYER_ELEM_ID} video'); "
            f"if(v){{v.currentTime={inicio}; v.play();}}\" "
            'style="display:block; margin: 2px 0;">'
            f"{etiqueta}</a>"
        )

    return "<div>" + "".join(filas) + "</div>"


def cargar_proveedores(api_url: str, tipo: str = "vlm") -> dict:
    """Consulta GET /api/v1/providers (VLM) o /api/v1/providers/llms (LLM) y devuelve proveedor -> lista de modelos."""
    ruta = "/api/v1/providers" if tipo == "vlm" else "/api/v1/providers/llms"
    try:
        resp = requests.get(f"{api_url.rstrip('/')}{ruta}", timeout=5)
        resp.raise_for_status()
        return resp.json().get("data", {}) or {}
    except requests.RequestException:
        return {}


def _primeras_opciones(proveedores_dict: dict):
    proveedores = list(proveedores_dict.keys())
    primer_proveedor = proveedores[0] if proveedores else None
    primeros_modelos = proveedores_dict.get(primer_proveedor, []) if primer_proveedor else []
    return proveedores, primer_proveedor, primeros_modelos


def refrescar_proveedores(api_url: str):
    vlm_dict = cargar_proveedores(api_url, "vlm")
    llm_dict = cargar_proveedores(api_url, "llm")

    vlm_proveedores, vlm_prov, vlm_modelos = _primeras_opciones(vlm_dict)
    llm_proveedores, llm_prov, llm_modelos = _primeras_opciones(llm_dict)

    return (
        vlm_dict,
        gr.update(choices=vlm_proveedores, value=vlm_prov),
        gr.update(choices=vlm_modelos, value=vlm_modelos[0] if vlm_modelos else None),
        llm_dict,
        gr.update(choices=llm_proveedores, value=llm_prov),
        gr.update(choices=llm_modelos, value=llm_modelos[0] if llm_modelos else None),
    )


def actualizar_modelos(proveedores_dict: dict, proveedor: str):
    modelos = proveedores_dict.get(proveedor, []) if proveedor else []
    return gr.update(choices=modelos, value=modelos[0] if modelos else None)


def mostrar_campos_por_tipo(tipo_analisis: str):
    es_eventos = tipo_analisis == TIPO_EVENTOS
    return (
        gr.update(visible=es_eventos),      # apply_alg
        gr.update(visible=not es_eventos),  # llm_provider
        gr.update(visible=not es_eventos),  # llm_model_name
    )


def iniciar_analisis(
    api_url: str,
    tipo_analisis: str,
    video_path: str,
    interval_time: float,
    user_prompt: str,
    vlm_provider: str,
    vlm_model_name: str,
    processing_mode_label: str,
    apply_alg: bool,
    llm_provider: str,
    llm_model_name: str,
):
    api_url = api_url.rstrip("/")

    if not video_path:
        yield "Debes subir un vídeo antes de iniciar el análisis.", None, None, ""
        return

    if not user_prompt:
        yield "Debes introducir una consulta/prompt para el análisis.", None, None, ""
        return

    if not vlm_provider or not vlm_model_name:
        yield "Selecciona un proveedor y un modelo VLM (usa 'Actualizar modelos disponibles' si la lista está vacía).", None, None, ""
        return

    processing_mode = ESTRATEGIAS.get(processing_mode_label)

    data = {
        "interval_time": interval_time,
        "user_prompt": user_prompt,
        "vlm_provider": vlm_provider,
        "vlm_model_name": vlm_model_name,
        "processing_mode": processing_mode,
    }

    if tipo_analisis == TIPO_EVENTOS:
        data["apply_alg"] = "true" if apply_alg else "false"
    else:
        if not llm_provider or not llm_model_name:
            yield "El resumen semántico requiere indicar proveedor y modelo del LLM.", None, None, ""
            return
        data["llm_provider"] = llm_provider
        data["llm_model_name"] = llm_model_name

    content_type, _ = mimetypes.guess_type(video_path)
    if not content_type or not content_type.startswith("video/"):
        content_type = "video/mp4"

    try:
        with open(video_path, "rb") as video_file:
            archivos = {"video": (os.path.basename(video_path), video_file, content_type)}
            respuesta = requests.post(
                f"{api_url}{_endpoint_analisis(tipo_analisis)}", data=data, files=archivos, timeout=60
            )
    except requests.RequestException as e:
        yield f"No se pudo conectar con la API en {api_url}: {e}", None, None, ""
        return

    if respuesta.status_code != 202:
        yield f"Error al iniciar el análisis ({respuesta.status_code}): {_extraer_detalle_error(respuesta)}", None, None, ""
        return

    project_id = respuesta.json()["data"]["project_id"]
    yield f"Análisis iniciado. ID de proyecto: {project_id}", None, project_id, ""

    status_url = f"{api_url}{_endpoint_status(project_id)}"
    resultados_url = f"{api_url}{_endpoint_resultados(tipo_analisis, project_id)}"

    while True:
        time.sleep(POLL_INTERVAL_SECONDS)

        try:
            status_resp = requests.get(status_url, timeout=10)
        except requests.RequestException as e:
            yield f"[{project_id}] Error consultando el estado: {e}", None, project_id, gr.update()
            continue

        if status_resp.status_code == 404:
            # el proyecto ya no existe: lo más probable es que se haya cancelado y
            # el pipeline haya borrado el directorio. Sondear en bucle no tiene sentido.
            yield f"[{project_id}] El proyecto ya no existe en el servidor (probablemente cancelado).", None, project_id, gr.update()
            return

        if status_resp.status_code != 200:
            yield f"[{project_id}] No se pudo consultar el estado ({status_resp.status_code}): {_extraer_detalle_error(status_resp)}", None, project_id, gr.update()
            continue

        status_data = status_resp.json()["data"]
        estado = status_data.get("state", "desconocido")
        progreso = status_data.get("progress", {})

        yield (
            f"[{project_id}] Estado: {estado} — frame {progreso.get('current_frame')}/{progreso.get('total_frames')}",
            None,
            project_id,
            gr.update(),
        )

        if estado == "completed":
            resultados_resp = requests.get(resultados_url, timeout=10)
            if resultados_resp.status_code != 200:
                # el fichero de resultados puede tardar un instante en aparecer justo tras completarse
                time.sleep(1)
                resultados_resp = requests.get(resultados_url, timeout=10)

            if resultados_resp.status_code == 200:
                datos_resultado = resultados_resp.json()["data"]
                # solo la detección de eventos devuelve intervalos con timestamps;
                # el resumen semántico devuelve un árbol de resumen, no hay nada que enlazar
                html_eventos = _construir_html_eventos(datos_resultado) if tipo_analisis == TIPO_EVENTOS else ""
                yield f"[{project_id}] Análisis completado.", datos_resultado, project_id, html_eventos
            else:
                yield (
                    f"[{project_id}] Terminó pero no se pudieron leer los resultados: {_extraer_detalle_error(resultados_resp)}",
                    None,
                    project_id,
                    gr.update(),
                )
            return

        if estado in ("error", "canceled"):
            yield f"[{project_id}] El análisis terminó con estado '{estado}'.", None, project_id, gr.update()
            return


def cancelar_analisis(api_url: str, project_id: str) -> str:
    if not project_id:
        return "No hay ningún análisis en curso para cancelar."

    try:
        respuesta = requests.post(f"{api_url.rstrip('/')}/api/v1/{project_id}/cancel", timeout=10)
    except requests.RequestException as e:
        return f"No se pudo conectar con la API: {e}"

    if respuesta.status_code == 200:
        return f"[{project_id}] Cancelación enviada correctamente."

    return f"[{project_id}] No se pudo cancelar ({respuesta.status_code}): {_extraer_detalle_error(respuesta)}"


with gr.Blocks(title="Demo VLM - Análisis de Vídeo") as demo:
    gr.Markdown(
        "# Análisis de Vídeo con VLM\n"
        "Sube un vídeo y obtén los intervalos de eventos detectados o un resumen semántico del contenido. "
        "Esta interfaz habla con la API por HTTP"
    )

    proveedores_state = gr.State({})
    proveedores_llm_state = gr.State({})
    project_id_state = gr.State(None)

    with gr.Row():
        api_url_input = gr.Textbox(label="URL de la API", value=DEFAULT_API_URL)
        refrescar_btn = gr.Button("Actualizar modelos disponibles")

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Vídeo a analizar", sources=["upload"], elem_id=VIDEO_PLAYER_ELEM_ID)
            tipo_analisis_input = gr.Radio(
                [TIPO_EVENTOS, TIPO_RESUMEN], value=TIPO_EVENTOS, label="Tipo de análisis"
            )
            processing_mode_input = gr.Radio(
                list(ESTRATEGIAS.keys()), value=list(ESTRATEGIAS.keys())[0], label="Estrategia de procesamiento"
            )
            interval_time_input = gr.Number(label="Intervalo entre frames (segundos)", value=2.0, minimum=0.1)
            user_prompt_input = gr.Textbox(
                label="Consulta / prompt", placeholder="Ej: Dime si aparece un perro en el vídeo"
            )

        with gr.Column():
            vlm_provider_input = gr.Dropdown(label="Proveedor VLM", choices=[])
            vlm_model_input = gr.Dropdown(label="Modelo VLM", choices=[])
            apply_alg_input = gr.Checkbox(label="Aplicar algoritmo de postprocesado", value=True)
            llm_provider_input = gr.Dropdown(label="Proveedor LLM (resumen)", choices=[], visible=False)
            llm_model_input = gr.Dropdown(label="Modelo LLM (resumen)", choices=[], visible=False)
            eventos_output = gr.HTML(label="Momentos detectados (pulsa para saltar en el vídeo)", sanitize_html=False)

    with gr.Row():
        iniciar_btn = gr.Button("Iniciar análisis", variant="primary")
        cancelar_btn = gr.Button("Cancelar análisis en curso")

    estado_output = gr.Textbox(label="Estado", interactive=False)
    resultados_output = gr.JSON(label="Resultados")

    salidas_refresco = [
        proveedores_state,
        vlm_provider_input,
        vlm_model_input,
        proveedores_llm_state,
        llm_provider_input,
        llm_model_input,
    ]
    demo.load(refrescar_proveedores, inputs=[api_url_input], outputs=salidas_refresco)
    refrescar_btn.click(refrescar_proveedores, inputs=[api_url_input], outputs=salidas_refresco)

    vlm_provider_input.change(
        actualizar_modelos, inputs=[proveedores_state, vlm_provider_input], outputs=[vlm_model_input]
    )
    llm_provider_input.change(
        actualizar_modelos, inputs=[proveedores_llm_state, llm_provider_input], outputs=[llm_model_input]
    )
    tipo_analisis_input.change(
        mostrar_campos_por_tipo,
        inputs=[tipo_analisis_input],
        outputs=[apply_alg_input, llm_provider_input, llm_model_input],
    )

    evento_analisis = iniciar_btn.click(
        iniciar_analisis,
        inputs=[
            api_url_input,
            tipo_analisis_input,
            video_input,
            interval_time_input,
            user_prompt_input,
            vlm_provider_input,
            vlm_model_input,
            processing_mode_input,
            apply_alg_input,
            llm_provider_input,
            llm_model_input,
        ],
        outputs=[estado_output, resultados_output, project_id_state, eventos_output],
    )
    cancelar_btn.click(cancelar_analisis, inputs=[api_url_input, project_id_state], outputs=[estado_output])
   
    cancelar_btn.click(fn=None, inputs=None, outputs=None, cancels=[evento_analisis])


if __name__ == "__main__":
    demo.launch()
