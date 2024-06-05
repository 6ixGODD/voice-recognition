import librosa
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment

plt.switch_backend('Agg')


def transform_audio(audio_path: str, output_path: str, fmt: str = "m4a"):
    audio = AudioSegment.from_file(audio_path)
    audio.export(output_path, format=fmt)


def plot_wave(audio_path: str, save_path: str, figsize=(2.56, 2.56), dpi=300):
    audio = AudioSegment.from_file(audio_path)
    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(audio.get_array_of_samples(), color='black')
    plt.xticks([])
    plt.yticks([])
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.axis('off')

    # Save the plot
    plt.savefig(save_path)
    plt.close()


def plot_spectrogram(audio_path: str, output_path: str):
    y, sr = librosa.load(audio_path)
    D = librosa.stft(y)
    D_db = librosa.amplitude_to_db(abs(D), ref=np.max)
    plt.figure(figsize=(6.4, 6.4), dpi=300)
    librosa.display.specshow(D_db, sr=sr, x_axis='time', y_axis='log')
    plt.xticks([])
    plt.yticks([])
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.axis('off')
    plt.savefig(output_path)
    plt.close()
