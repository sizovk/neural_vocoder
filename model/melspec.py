from dataclasses import dataclass

import torch
from torch import nn

import torchaudio

import librosa  


@dataclass
class MelSpectrogramConfig:
    sr: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min: int = 0
    f_max: int = 8000
    n_mels: int = 80
    power: float = 1.0

    # value of melspectrograms if we fed a silence into `MelSpectrogram`
    pad_value: float = -11.5129251


class MelSpectrogram(nn.Module):

    def __init__(self, **kwargs):
        super(MelSpectrogram, self).__init__()

        self.config = MelSpectrogramConfig(**kwargs)

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sr,
            win_length=self.config.win_length,
            hop_length=self.config.hop_length,
            n_fft=self.config.n_fft,
            f_min=self.config.f_min,
            f_max=self.config.f_max,
            n_mels=self.config.n_mels
        )

        # The is no way to set power in constructor in 0.5.0 version.
        self.mel_spectrogram.spectrogram.power = self.config.power

        # Default `torchaudio` mel basis uses HTK formula. In order to be compatible with WaveGlow
        # we decided to use Slaney one instead (as well as `librosa` does by default).
        mel_basis = librosa.filters.mel(
            sr=self.config.sr,
            n_fft=self.config.n_fft,
            n_mels=self.config.n_mels,
            fmin=self.config.f_min,
            fmax=self.config.f_max
        ).T
        self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(mel_basis))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        :param audio: Expected shape is [B, T]
        :return: Shape is [B, n_mels, T']
        """

        mel = self.mel_spectrogram(audio) \
            .clamp_(min=1e-5) \
            .log_()

        return mel