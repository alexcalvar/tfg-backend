from llama_cpp.llama_chat_format import Llava15ChatHandler, Qwen25VLChatHandler

VLM_HANDLERS = {
    "llava": Llava15ChatHandler,
    "qwen2.5": Qwen25VLChatHandler
}