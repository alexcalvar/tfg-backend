from enum import Enum

class VLMProvider(Enum):
    LLAMACPP = "llamacpp"
    GOOGLE = "google"
    OLLAMA = "ollama"
    OPENROUTE = "openroute"

class LLMProvider(Enum):
    LLAMACPP = "llamacpp"
    GOOGLE = "google"
    OLLAMA = "ollama"
    OPENROUTE = "openroute"
    GROQ = "groq"


class NormalizerAlgorithm(Enum):
    SLIDINGWINDOW = "sliding_window"

class ParserType(Enum):
    JSON = "json"
    TEXT = "text"

class StrategyType(Enum):
    BATCH = "batch_strategy"
    TEMPORAL = "temporal_strategy" 


class PostProcessingStrategy(Enum):
    pass