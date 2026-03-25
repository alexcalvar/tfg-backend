from enum import Enum

class ParserType(Enum):
    JSON = "json"
    TEXT = "text"

class StrategyType(Enum):
    BATCH = "batch_strategy"
    TEMPORAL = "temporal_strategy"