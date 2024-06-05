import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import shutil


def m4a2wav(m4a_path, wav_path):
    from pydub import AudioSegment
    audio = AudioSegment.from_file(m4a_path)
    audio.export(wav_path, format="wav")


def audio2spectrogram(audio_path, save_path):
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
    plt.savefig(save_path)
    plt.close()


def run():
    audio_dir = "data/Audio"
    wave_dir = "data/Wave"
    if not os.path.exists(wave_dir):
        os.makedirs(wave_dir)

    for i, d in enumerate(os.listdir(audio_dir)):
        if not os.path.exists(os.path.join(wave_dir, d)):
            os.makedirs(os.path.join(wave_dir, d))
        for j, f in enumerate(os.listdir(os.path.join(audio_dir, d))):
            if f.endswith(".m4a"):
                print(f"Processing {os.path.join(audio_dir, d, f)}")
                m4a2wav(os.path.join(audio_dir, d, f), os.path.join(wave_dir, d, f"{j}.wav"))
            elif f.endswith(".wav"):
                print(f"Processing {os.path.join(audio_dir, d, f)}")
                shutil.copy(os.path.join(audio_dir, d, f), os.path.join(wave_dir, d, f"{j}.wav"))
            else:
                print(f"Unsupported format {f}")
                continue

    output_dir = "data/SpectrogramImages"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, d in enumerate(os.listdir(wave_dir)):
        if not os.path.exists(os.path.join(output_dir, d)):
            os.makedirs(os.path.join(output_dir, d))
        if d not in ["xiang", "yaobing"]:
            continue  # skip
        for j, f in enumerate(os.listdir(os.path.join(wave_dir, d))):
            print(f"Processing {os.path.join(wave_dir, d, f)}")
            audio2spectrogram(os.path.join(wave_dir, d, f), os.path.join(output_dir, d, f"{j}.png"))


if __name__ == '__main__':
    run()
