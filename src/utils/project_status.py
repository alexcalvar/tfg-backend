from enum import Enum

class ProjectStatus(str, Enum):
    QUEUED = "queued"
    EXTRACTING = "extracting_frames"
    ANALYZING = "analyzing_frames"
    COMPLETED = "completed"
    ERROR = "error"

