import re
import sys
import time
from pathlib import Path
from typing import Tuple

import sounddevice as sd
from PyQt5.QtCore import QSize, QTimer
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from scipy.io.wavfile import write

from utils.logger import get_logger


# noinspection PyUnresolvedReferences
class CollectorApp(QWidget):
    def __init__(
            self,
            dataset_dir: str = 'dataset',  # Directory to save audio files
            record_duration: Tuple[int, int] = (10, 20),  # Minimum and maximum duration in seconds
            record_fs: int = 44100,  # Audio sampling frequency in Hz
            audio_channels: int = 1
    ):
        super().__init__()
        self._logger = get_logger("CollectorApp", "logs/collector.log")
        self._logger.info("Starting Collector App")
        self._dataset_dir = Path(dataset_dir)
        self._username = None

        # Audio settings and variables
        self.__current_duration = 0
        self._audio = None
        self._record_duration = record_duration
        self._record_fs = record_fs
        self._audio_channels = audio_channels

        self.layout = QVBoxLayout()

        # First page setup
        self.first_page = QWidget()
        self.first_page_input_layout = QHBoxLayout()
        self.first_page_layout = QVBoxLayout()

        # Name input
        self.name_edit = QLineEdit(self)
        self.name_edit.setPlaceholderText("Enter your name")
        self.name_edit.setStyleSheet("color: grey; font-size: 16px; font-family: Arial;")

        # Next button
        self.next_button = QPushButton("Next")
        self.next_button.setFont(QFont("Arial", 10))
        self.next_button.clicked.connect(self.__next_page)

        # Error label for checking if name is valid
        self.error_label = QLabel()
        self.error_label.setStyleSheet("color: red; font-size: 12px; font-family: Arial;")
        self.error_label.setHidden(True)

        # Setup layout
        self.first_page_input_layout.addWidget(self.name_edit)
        self.first_page_input_layout.addWidget(self.next_button)
        self.first_page_layout.addLayout(self.first_page_input_layout)
        self.first_page_layout.addWidget(self.error_label)
        self.first_page.setLayout(self.first_page_layout)

        # Second page setup
        self.second_page = QWidget()
        self.second_page_control_layout = QHBoxLayout()
        self.second_page_summit_layout = QHBoxLayout()
        self.second_page_layout = QVBoxLayout()

        # Record button
        self.record_button = QPushButton()
        self.record_button.setFixedSize(QSize(40, 40))
        self.record_button.setIcon(QIcon('asserts/svg/play.svg'))
        self.record_button.clicked.connect(self.__start_recording)

        # Record progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(self._record_duration[1] * 1000)  # Convert to milliseconds
        self.progress_bar.setFont(QFont("Arial", 10))
        self.progress_bar.setTextVisible(False)

        # Record duration label
        self.duration_label = QLabel("0s")
        self.duration_label.setFont(QFont("Arial", 10))
        self.duration_label.setStyleSheet("color: grey;")
        self.duration_label.setFixedWidth(30)

        # Submit button
        self.submit_button = QPushButton("Submit")
        self.submit_button.setEnabled(False)
        self.submit_button.setFont(QFont("Arial", 10))
        self.submit_button.clicked.connect(self.__summit_audio)

        # Timer for recording duration
        self.timer = QTimer()
        self.timer.timeout.connect(self.__update_progress)

        # Setup layout
        self.second_page_control_layout.addWidget(self.record_button)
        self.second_page_control_layout.addWidget(self.progress_bar)
        self.second_page_control_layout.addWidget(self.duration_label)
        self.second_page_summit_layout.addStretch(1)
        self.second_page_summit_layout.addWidget(self.submit_button)
        self.second_page_layout.addLayout(self.second_page_control_layout)
        self.second_page_layout.addLayout(self.second_page_summit_layout)
        self.second_page.setLayout(self.second_page_layout)

        # Main window setup
        self.layout.addWidget(self.first_page)
        self.setLayout(self.layout)
        self.setWindowTitle("Collector App")
        self.setGeometry(300, 300, 400, 120)
        self.show()

    def __next_page(self):
        username = self.name_edit.text()
        if not username:
            self.error_label.setText("Name cannot be empty")
            self.error_label.setHidden(False)
            return
        if not re.match(r"^[a-zA-Z0-9_]*$", username):
            self.error_label.setText("Name should contain only letters, numbers, and underscores")
            self.error_label.setHidden(False)
            return
        self._username = username.lower()
        (self._dataset_dir / self._username).mkdir(parents=True, exist_ok=True)
        self._logger.info(f"User {self._username} is ready to record audio")
        self.error_label.setHidden(True)
        self.layout.removeWidget(self.first_page)
        self.first_page.deleteLater()
        self.layout.addWidget(self.second_page)

    def __start_recording(self):
        if self.timer.isActive():
            self._logger.info("Stopped recording audio")
            self.record_button.setIcon(QIcon('asserts/svg/play.svg'))
            self.timer.stop()
            self.record_button.setEnabled(True)  # Re-enable button after recording
            self.submit_button.setEnabled(True)
        else:
            self._logger.info("Recording audio...")
            self._audio = sd.rec(
                int(self._record_duration[1] * self._record_fs),
                samplerate=self._record_fs,
                channels=self._audio_channels,
                dtype='float32'
            )
            self.record_button.setIcon(QIcon('asserts/svg/stop.svg'))
            self.record_button.setEnabled(False)
            self.submit_button.setEnabled(False)
            self.__current_duration = 0
            self.timer.start(50)

    def __summit_audio(self):
        audio_path = self._dataset_dir / self._username / f"{self._username}.{int(time.time())}.wav"
        write(str(audio_path), self._record_fs, self._audio)
        self._logger.info(f"Saved audio at {audio_path}")
        self.submit_button.setEnabled(False)
        self.record_button.setEnabled(True)
        self.record_button.setIcon(QIcon('asserts/svg/play.svg'))
        self.timer.stop()
        self.duration_label.setText("submitted!")

    def __update_progress(self):
        self.__current_duration += 0.05
        self.progress_bar.setValue(int(self.__current_duration * 1000))
        if self.__current_duration < self._record_duration[0]:
            self.duration_label.setText(f"{int(self.__current_duration)}s")
        elif self._record_duration[0] <= self.__current_duration < self._record_duration[1]:
            self.duration_label.setText(f"{int(self.__current_duration)}s")
            self.record_button.setEnabled(True)
        else:
            self.timer.stop()
            self.record_button.setEnabled(True)
            self.submit_button.setEnabled(True)
            self.__current_duration = 0  # Reset timer for next recording
            self.duration_label.setText("completed!")
            self.record_button.setIcon(QIcon('asserts/svg/play.svg'))


if __name__ == '__main__':
    import traceback

    try:
        app = QApplication(sys.argv)
        ex = CollectorApp()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)
