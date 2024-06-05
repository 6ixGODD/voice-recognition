import os

import matplotlib.pyplot as plt

plt.switch_backend('Agg')


def audio2wave_graph(audio_path, save_path):
    from pydub import AudioSegment
    audio = AudioSegment.from_file(audio_path)
    array = audio.get_array_of_samples()
    # print(array)
    plt.figure(figsize=(6.4, 6.4), dpi=300)
    plt.plot(array, color='black')
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


def run():
    root_dir = "data/_temp"
    output_dir = "data/WaveGraphImages"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, d in enumerate(os.listdir(root_dir)):
        if not os.path.exists(os.path.join(output_dir, d)):
            os.makedirs(os.path.join(output_dir, d))
        for j, f in enumerate(os.listdir(os.path.join(root_dir, d))):
            print(f"Processing {os.path.join(root_dir, d, f)}")
            audio2wave_graph(os.path.join(root_dir, d, f), os.path.join(output_dir, d, f"{j}.png"), fmt="wav")


if __name__ == '__main__':
    run()
