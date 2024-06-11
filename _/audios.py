from utils.audios import combine_audio
from pathlib import Path

Path('output/audios').mkdir(parents=True, exist_ok=True)
combine_audio('../data/audios/bochen', 'output/audios/combined-bochen.wav')
