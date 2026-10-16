"""Shared waveform-to-log-mel frontend for ESC-50."""

import torch
import torchaudio

from src import config


class LogMelFrontend(torch.nn.Module):
    """Convert a 32 kHz waveform into a log-mel spectrogram."""

    def __init__(self) -> None:
        super().__init__()

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.SR,
            n_fft=config.N_FFT,
            win_length=config.WIN_LENGTH,
            hop_length=config.HOP,
            f_min=config.FMIN,
            f_max=config.FMAX,
            n_mels=config.N_MELS,
            power=2.0,
            center=True,
            pad_mode="reflect",
            norm=None,
            mel_scale="htk",
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: Audio tensor shaped (samples,) or (batch, samples).

        Returns:
            Log-mel tensor shaped (batch, n_mels, time_frames).
        """
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        if waveform.ndim != 2:
            raise ValueError(
                "Expected waveform shape (samples,) or (batch, samples); "
                f"received {tuple(waveform.shape)}."
            )

        if waveform.shape[-1] != config.N_SAMPLES:
            raise ValueError(
                f"Expected {config.N_SAMPLES} samples "
                f"({config.DURATION_S}s at {config.SR} Hz), "
                f"received {waveform.shape[-1]}."
            )

        if not waveform.is_floating_point():
            waveform = waveform.float()

        mel = self.mel_spectrogram(waveform)
        log_mel = torch.log(mel.clamp_min(config.EPS))

        return log_mel
