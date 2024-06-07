import logging
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
from PIL import Image
import numpy as np
import pyqtgraph as pg
import sounddevice as sd
from PyQt5.QtCore import pyqtSignal, QSize, Qt, QThread, QTimer
from PyQt5.QtGui import QFont, QIcon, QPixmap
from PyQt5.QtWidgets import QApplication, QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget
from scipy.io.wavfile import write

from models.cnn import ConvolutionNeuralNetworkClassifierBackend
from models.lbp import LocalBinaryPatternsClassifierBackend
from utils.audios import plot_spectrogram
from utils.images import calculate_lbp_vector, faster_calculate_lbp
from utils.logger import get_logger

PLAY_ICON = QIcon('applications/asserts/svg/play.svg')
STOP_ICON = QIcon('applications/asserts/svg/stop.svg')


# noinspection PyUnresolvedReferences
class AnalyserApp(QWidget):
    def __init__(
            self, categories: Dict[int, str],
            image_size: Tuple[int, int],
            temp_dir: str = "_temp",
            sample_rate: int = 44100
    ):
        super().__init__()
        self._logger = get_logger("AnalyserApp", "_logs/analyser.log")
        self._logger.info("Starting Analyser App")
        self._temp_dir = Path(temp_dir)
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._sample_rate = sample_rate
        self._categories = categories
        self._image_size = image_size

        # Estimators settings
        self._estimators: List[str] = []  # Store the names of the estimators
        self.lbp_based_estimators: Dict[str, LocalBinaryPatternsClassifierBackend] = {}
        self.cnn_based_estimators: Dict[str, Tuple[ConvolutionNeuralNetworkClassifierBackend, Callable]] = {}

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Top row
        self.top_layout = QHBoxLayout()
        self.control_button = QPushButton()
        self.control_button.setFixedSize(QSize(40, 40))
        self.control_button.setIcon(PLAY_ICON)
        self.control_button.clicked.connect(self.__control)
        self.top_layout.addWidget(self.control_button)

        self.audio_waveform = pg.PlotWidget()
        self.audio_waveform.setYRange(-1, 1)
        self.audio_waveform.setBackground('w')
        # Remove axis
        self.audio_waveform.getPlotItem().hideAxis('left')
        self.audio_waveform.getPlotItem().hideAxis('bottom')
        self.audio_waveform.getPlotItem().hideAxis('right')
        self.audio_waveform.getPlotItem().hideAxis('top')
        self.audio_waveform.setFixedHeight(60)
        self.audio_waveform.setFixedWidth(640)
        self.audio_waveform.getPlotItem().hideButtons()
        self.audio_waveform.setMouseEnabled(x=False, y=False)
        self.audio_waveform.getPlotItem().getViewBox().setMenuEnabled(False)
        self.top_layout.addWidget(self.audio_waveform)
        self.layout.addLayout(self.top_layout)

        # Middle row
        self.middle_layout = QVBoxLayout()

        self.label_display = QLabel("Prediction:")
        self.label_display.setFont(QFont("Times New Roman", 10))
        self.label_display.setAlignment(Qt.AlignCenter)
        self.label_display.setFixedHeight(40)
        self.middle_layout.addWidget(self.label_display)

        self.image_display = QLabel()
        self.image_display.setFont(QFont("Arial", 10))
        self.image_display.setFixedSize(QSize(640, 640))
        self.image_display.setAlignment(Qt.AlignCenter)
        self.middle_layout.addWidget(self.image_display)

        self.layout.addLayout(self.middle_layout)

        # Bottom row
        self.bottom_layout = QHBoxLayout()
        self.bottom_layout.addStretch(1)

        self.model_selector = QComboBox()
        self.model_selector.setFixedWidth(300)
        self.model_selector.setFont(QFont("Times New Roman", 10))
        self.model_selector.addItems(self._estimators)
        self.bottom_layout.addWidget(self.model_selector)

        self.exit_button = QPushButton("Exit")
        self.exit_button.setFont(QFont("Arial", 10))
        self.exit_button.clicked.connect(self.close)
        self.bottom_layout.addWidget(self.exit_button)

        self.layout.addLayout(self.bottom_layout)

        # Audio settings
        self.wave_timer = QTimer()
        self.wave_timer.timeout.connect(self.__update_waveform)
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.__update_image_display)
        self.wave_stream = sd.InputStream(
            callback=self.__wave_callback, channels=1, samplerate=self._sample_rate, blocksize=1024
        )
        self.wave_data = np.zeros(1024)
        self.wave_data_line = self.audio_waveform.plot(self.wave_data, pen=pg.mkPen('b', width=1))
        self.sample_data = np.zeros(self._sample_rate * 10)

        # Threads
        self.spectrogram_thread = SpectrogramThread(
            np.zeros(self._sample_rate * 10), str(self._temp_dir), sr=self._sample_rate
        )
        self.spectrogram_thread.finished.connect(self.update_timer.start)
        self.classification_thread = None

        self.setWindowTitle("Analyser App")
        self.setGeometry(300, 300, 640, 800)

    def register_lbp_estimator(self, name: str, estimator: LocalBinaryPatternsClassifierBackend):
        if name in self.lbp_based_estimators:
            raise ValueError(f"Estimator with name '{name}' already exists")
        self.lbp_based_estimators[name] = estimator
        for dist, _ in estimator.distances.items():
            self._estimators.append(f"{name} (distance-based: {dist})")
        if len(estimator.estimators):
            for key in estimator.estimators:
                self._estimators.append(f"{name} ({key})")
        self.model_selector.addItems(self._estimators)
        self.__update_combobox()
        self._logger.info(f"Registered LBP-based estimator: {name}")

    def register_cnn_estimator(self, name: str, estimator: ConvolutionNeuralNetworkClassifierBackend, tfs: Callable):
        if name in self.cnn_based_estimators:
            raise ValueError(f"Estimator with name '{name}' already exists")
        self.cnn_based_estimators[name] = estimator, tfs
        self._estimators.append(f"{name} (CNN-based)")
        self.model_selector.addItems(self._estimators)
        self.__update_combobox()
        self._logger.info(f"Registered CNN-based estimator: {name}")

    def __control(self):
        if not self.wave_timer.isActive():
            self.wave_stream.start()
            self.wave_timer.start(50)
            self.update_timer.start(12000)
            self.control_button.setIcon(STOP_ICON)
        else:
            self.wave_stream.stop()
            self.wave_timer.stop()
            self.update_timer.stop()
            self.control_button.setIcon(PLAY_ICON)

    def __update_waveform(self):
        self.wave_data_line.setData(self.wave_data)

    def __wave_callback(self, indata, frames, _, status):
        if status:
            self._logger.error(status)
        self.wave_data = indata[:, 0]
        self.sample_data = np.roll(self.sample_data, -frames)  # Shift data to the left
        self.sample_data[-frames:] = indata[:, 0]

    def __update_image_display(self):
        self.update_timer.stop()
        audio = self.sample_data[-self._sample_rate * 10:]
        self.__generate_spectrogram(audio)
        pix_map = QPixmap(str(self._temp_dir / 'temp.jpg'))
        pix_map = pix_map.scaled(self.image_display.size(), Qt.KeepAspectRatio)
        self.image_display.setPixmap(pix_map)
        if self.classification_thread is not None:
            self.classification_thread.terminate()  # Terminate previous thread
        if self.model_selector.currentText().endswith("(CNN-based)"):  # CNN-based estimator
            self.classification_thread = ClassificationThread(
                self._logger,
                str(self._temp_dir / 'temp.jpg'),
                self.cnn_based_estimators[self.model_selector.currentText().split(" (")[0]]
            )
        else:  # LBP-based estimator
            self.classification_thread = ClassificationThread(
                self._logger,
                str(self._temp_dir / 'temp.jpg'),
                self.lbp_based_estimators[self.model_selector.currentText().split(" (")[0]],
                self.model_selector.currentText().split(" (")[1].split(")")[0]
            )
        self.classification_thread.finished.connect(self.__update_label_display)
        self.classification_thread.start()

    def __update_label_display(self, prediction: int):
        self.label_display.setText(f"Prediction: {self._categories[prediction]}")

    def __update_combobox(self):
        self.model_selector.clear()
        self.model_selector.addItems(self._estimators)

    def __generate_spectrogram(self, audio: np.ndarray):
        self.spectrogram_thread.audio = audio
        self.spectrogram_thread.start()

    def close(self):
        self.wave_stream.stop()
        self.wave_timer.stop()
        self.update_timer.stop()
        self.spectrogram_thread.terminate()
        if self.classification_thread is not None:
            self.classification_thread.terminate()
        for file in self._temp_dir.glob('*'):
            file.unlink()
        self._temp_dir.rmdir()
        self._logger.info("Exiting Analyser App")
        super().close()


# noinspection PyUnresolvedReferences
class SpectrogramThread(QThread):
    finished = pyqtSignal()

    def __init__(self, audio: np.ndarray, output_dir: str, sr: int = 44100):
        super().__init__()
        self.audio = audio
        self.output_path = output_dir
        self.sr = sr

    def run(self):
        # Save audio to temporary file
        write(str(Path(self.output_path) / 'temp.wav'), self.sr, self.audio)
        plot_spectrogram(str(Path(self.output_path) / 'temp.wav'), str(Path(self.output_path) / 'temp.jpg'))
        self.finished.emit()


# noinspection PyUnresolvedReferences
class ClassificationThread(QThread):
    finished = pyqtSignal(int)

    def __init__(
            self,
            logger: logging.Logger,
            image_path: str,
            estimator: [
                Union[LocalBinaryPatternsClassifierBackend, Tuple[ConvolutionNeuralNetworkClassifierBackend, Callable]]
            ],
            name: Optional[str] = None
    ):
        super().__init__()
        self._logger = logger
        self.image_path = image_path
        self.estimator = estimator
        self.name = name

    def run(self):
        if self.name is None:  # CNN-based estimator
            self._logger.info("Predicting using CNN-based estimator")
            clf, tfs = self.estimator
            image = Image.open(self.image_path)
            image = tfs(image)
            image = image[None]
            prediction = clf.predict(image)[0]
        elif "distance-based" in self.name:  # LBP-based estimator with distance function
            image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
            lbp = faster_calculate_lbp(image)
            v = calculate_lbp_vector(lbp)
            self._logger.info(f"Predicting using LBP-based estimator with distance function: {self.name}")
            name, dist = self.name.split(": ")
            prediction = self.estimator.predict(v, estimator=name, distance_func=dist)
        else:  # LBP-based estimator
            image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
            lbp = faster_calculate_lbp(image)
            v = calculate_lbp_vector(lbp)
            self._logger.info(f"Predicting using LBP-based estimator: {self.name}")
            prediction = self.estimator.predict(v, estimator=self.name)
        self.finished.emit(prediction)


if __name__ == "__main__":
    import traceback
    from datasets.lbp import LocalBinaryPatternsImageClassificationDataset
    from models.lbp import LocalBinaryPatternsClassifierBackend
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from torchvision import transforms

    DATASET_DIR = "data/augmented"
    ESTIMATORS = {
        'SVM: Linear, C=1.0': SVC(kernel='linear', C=1.0),
        'Random Forest: n_estimators=100': RandomForestClassifier(n_estimators=100),
    }

    MODEL_NAME = "resnet18"
    WEIGHT = 'output/resnet-18-voice-reco/best_resnet18_tongue.pth'
    NUM_CLASSES = 13
    TRANSFORMS = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

    try:
        dataset = LocalBinaryPatternsImageClassificationDataset()
        dataset.load_images(root=DATASET_DIR)
        lbp_classifier = LocalBinaryPatternsClassifierBackend(
            estimators=ESTIMATORS
        )
        lbp_classifier.train(dataset)

        cnn_classifier = ConvolutionNeuralNetworkClassifierBackend(
            weight_path=WEIGHT, model_name=MODEL_NAME
        )
        cnn_classifier.init_model(num_classes=NUM_CLASSES)

        app = QApplication(sys.argv)
        window = AnalyserApp(
            dataset.categories, (dataset.lbp_images[0].shape[0], dataset.lbp_images[0].shape[1])
        )
        window.register_lbp_estimator("LBP", lbp_classifier)
        window.register_cnn_estimator("ResNet-18", cnn_classifier, TRANSFORMS)
        window.show()
        app.exec_()
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)
