from llama_cpp import Llama

# Usamos la ruta completa y absoluta con la doble barra de Windows
ruta_absoluta = "C:\\Users\\Usuario\\Documents\\IngenieriaInformatica\\4-CURSO\\TFG\\TFG-Backend\\models\\Qwen2-VL-7B-Instruct-Q4_K_M.gguf"

print(f"🔍 Intentando cargar el modelo desde:\n{ruta_absoluta}\n")

try:
    # Llamamos a Llama directamente, SIN LangChain, y con verbose=True para ver el error real
    llm = Llama(
        model_path=ruta_absoluta,
        verbose=True
    )
    print("\n✅ ¡CARGA EXITOSA! El archivo y el motor están perfectos.")
except Exception as e:
    print("\n❌ FALLO AL CARGAR. Aquí está el error real:")
    print(e)