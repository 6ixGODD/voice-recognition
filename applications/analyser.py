from PyQt5.QtWidgets import QWidget

from utils.logger import get_logger

class AnalyserApp(QWidget):
    def __init__(self):
        super().__init__()
        self._logger = get_logger("AnalyserApp", "logs/analyser.log")
