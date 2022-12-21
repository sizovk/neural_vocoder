import os
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

class LJSpeechDataset(Dataset):
    def __init__(self, wavs_dir="./data/LJSpeech-1.1/wavs"):
        self.path_wavs = [os.path.join(wavs_dir, wav) for wav in sorted(os.listdir(wavs_dir))]

    def __len__(self):
        return len(self.path_wavs)

    def __getitem__(self, idx):
        wav, sr = librosa.load(self.path_wavs[idx])
        wav = torch.from_numpy(wav)
        return wav
