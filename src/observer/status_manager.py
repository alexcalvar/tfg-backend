import os
import datetime

from src.utils.project_status import ProjectStatus
from src.utils.file_utils import save_json

from src.observer.observer import StatusObserver

class ProjectStatusManager(StatusObserver): 
    
    def __init__(self, base_run_dir: str):
        self.status_path = os.path.join(base_run_dir, "status.json")
        self.total_frames = 0
    
    
    def update_status(self, state: ProjectStatus, message: str, current_frame: int):
        status_data = {
            "state": state.value, 
            "message": message,
            "progress": {
                "current_frame": current_frame,
                "total_frames": self.total_frames
            },
            "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        save_json(status_data, self.status_path)