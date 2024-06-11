import librosa as _lb
import matplotlib.pyplot as _plt
import numpy as _np
from pathlib import Path
from pydub import AudioSegment as _AudioSegment


def combine_audio(audio_dir: str, output_path: str):
    audio = _AudioSegment.empty()
    for file in Path(audio_dir).iterdir():
        audio += _AudioSegment.from_file(str(file))
    audio.export(output_path, format="wav")


def transform_audio(audio_path: str, output_path: str, fmt: str = "wav"):
    audio = _AudioSegment.from_file(audio_path)
    audio.export(output_path, format=fmt)


def plot_wave(audio_path: str, save_path: str, figsize=(2.56, 2.56), dpi=300):
    audio = _AudioSegment.from_file(audio_path)
    _plt.figure(figsize=figsize, dpi=dpi)
    _plt.plot(audio.get_array_of_samples(), color='black')
    _plt.xticks([])
    _plt.yticks([])
    _plt.gca().xaxis.set_major_locator(_plt.NullLocator())
    _plt.gca().yaxis.set_major_locator(_plt.NullLocator())
    _plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    _plt.margins(0, 0)
    _plt.gca().xaxis.set_major_locator(_plt.NullLocator())
    _plt.gca().yaxis.set_major_locator(_plt.NullLocator())
    _plt.axis('off')

    # Save the plot
    _plt.savefig(save_path)
    _plt.close()


def plot_spectrogram(audio: str, output_path: str):
    y, sr = _lb.load(audio)
    D = _lb.stft(y)
    D_db = _lb.amplitude_to_db(abs(D), ref=_np.max)
    _plt.figure(figsize=(6.4, 6.4), dpi=300)
    _lb.display.specshow(D_db, sr=sr, x_axis='time', y_axis='log')
    _plt.xticks([])
    _plt.yticks([])
    _plt.gca().xaxis.set_major_locator(_plt.NullLocator())
    _plt.gca().yaxis.set_major_locator(_plt.NullLocator())
    _plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    _plt.margins(0, 0)
    _plt.gca().xaxis.set_major_locator(_plt.NullLocator())
    _plt.gca().yaxis.set_major_locator(_plt.NullLocator())
    _plt.axis('off')
    _plt.savefig(output_path)
    _plt.close()
