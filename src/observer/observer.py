from abc import ABC, abstractmethod
from typing import List

from src.utils.project_status import ProjectStatus

class StatusObserver(ABC):
    
    @abstractmethod
    def update_status(self, state: ProjectStatus, message: str, current_frame: int):
        pass


class StatusObservable(ABC):
    def __init__(self):
        self._observers: List[StatusObserver] = []

    def attach(self, observer: StatusObserver):
        """Añade un suscriptor"""
        if observer not in self._observers:
            self._observers.append(observer)

    def detach(self, observer: StatusObserver):
        """Elimina un suscriptor"""
        self._observers.remove(observer)

    def notify(self, state: ProjectStatus, message: str, current_frame: int = 0):
        """Avisa a todos los suscriptores del nuevo estado"""
        for observer in self._observers:
            observer.update_status(state, message, current_frame)